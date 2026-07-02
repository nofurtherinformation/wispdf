/**
 * Conformance tests for the hash kernel family (Agent D).
 *
 * Reads `tests/conformance/fixtures/relational.json`, packs buffers per
 * README §2, and exercises every case on BOTH scalar and SIMD builds.
 *
 * Fixture protocol:
 * - property `"equal_inputs_equal_hashes"`: verify that rows in the same
 *   `expected.equal_groups` group all produce identical i64 hashes.
 * - property `"group_partition"` on group_build cases: verify
 *   `group_count` and partition structure (set-of-sets, unordered).
 * - `"deterministic_in_place"` on hash_combine: call twice with identical
 *   inputs and verify outputs match.
 * - join cases: verify pair list in exact order.
 *
 * Retry protocol tests (in "retry path" describe block at the end):
 * - Force tiny htCap to exercise the -1 grow path.
 * - Force tiny outCap to exercise the n>outCap grow path.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { detectSimd } from '../../../src/memory/loader.js';
import {
  groupBuild,
  joinHashInner,
  joinHashLeft,
  type HashExports,
} from '../../../src/kernels/hash/index.js';

// ---------------------------------------------------------------------------
// Fixture loading
// ---------------------------------------------------------------------------

const __dir = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = join(__dir, '../../conformance/fixtures/relational.json');
const WASM_DIR = fileURLToPath(new URL('../../../wasm/dist/', import.meta.url));

interface FixtureCase {
  export: string;
  name: string;
  note?: string;
  property?: string;
  inputs: Record<string, unknown>;
  expected?: Record<string, unknown>;
}

let fixture: { cases: FixtureCase[] };

// ---------------------------------------------------------------------------
// Build config
// ---------------------------------------------------------------------------

const BUILDS: ReadonlyArray<{ label: string; simd: boolean }> = [
  { label: 'scalar', simd: false },
  { label: 'simd', simd: true },
];

// ---------------------------------------------------------------------------
// WASM module loader — returns ALL exports (not just Phase-1 memory core)
// ---------------------------------------------------------------------------

/**
 * Load the WASM binary directly and return the full exports object.
 * Unlike `loadWasmModule`, this exposes both Phase-1 (alloc/free/memory) and
 * Phase-2 (hash_i32, group_build, …) exports.
 */
async function loadAllExports(simd: boolean): Promise<HashExports> {
  const fileName = simd ? 'simd.wasm' : 'scalar.wasm';
  const filePath = join(WASM_DIR, fileName);
  const bytes = await readFile(filePath);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports as unknown as HashExports;
}

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------

/** Pack a `0/1` validity array into Arrow-LSB bytes. */
function packBitmap(vals: number[], buf: Uint8Array): void {
  buf.fill(0);
  for (let i = 0; i < vals.length; i++) {
    if (vals[i]) buf[i >> 3]! |= 1 << (i & 7);
  }
}

/** Allocate and write an i64 BigInt array from decimal strings. Returns ptr. */
function allocI64(ex: HashExports, strs: string[]): number {
  if (strs.length === 0) return ex.alloc(0);
  const ptr = ex.alloc(strs.length * 8);
  if (ptr === 0) throw new Error('OOM: i64 alloc');
  const view = new BigInt64Array(ex.memory.buffer, ptr, strs.length);
  for (let i = 0; i < strs.length; i++) {
    view[i] = BigInt(strs[i]!);
  }
  return ptr;
}

/** Allocate and write an i32 array. Returns ptr. */
function allocI32(ex: HashExports, vals: number[]): number {
  if (vals.length === 0) return ex.alloc(0);
  const ptr = ex.alloc(vals.length * 4);
  if (ptr === 0) throw new Error('OOM: i32 alloc');
  new Int32Array(ex.memory.buffer, ptr, vals.length).set(vals);
  return ptr;
}

/** Allocate and write a u32 array. Returns ptr. */
function allocU32(ex: HashExports, vals: number[]): number {
  if (vals.length === 0) return ex.alloc(0);
  const ptr = ex.alloc(vals.length * 4);
  if (ptr === 0) throw new Error('OOM: u32 alloc');
  new Uint32Array(ex.memory.buffer, ptr, vals.length).set(vals);
  return ptr;
}

/** Allocate and write an f64 array. Returns ptr. */
function allocF64(ex: HashExports, vals: (number | string)[]): number {
  if (vals.length === 0) return ex.alloc(0);
  const ptr = ex.alloc(vals.length * 8);
  if (ptr === 0) throw new Error('OOM: f64 alloc');
  const view = new Float64Array(ex.memory.buffer, ptr, vals.length);
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i];
    if (v === 'NaN') view[i] = NaN;
    else if (v === 'Infinity') view[i] = Infinity;
    else if (v === '-Infinity') view[i] = -Infinity;
    else view[i] = v as number;
  }
  return ptr;
}

/** Allocate and write an f32 array. Returns ptr. */
function allocF32(ex: HashExports, vals: (number | string)[]): number {
  if (vals.length === 0) return ex.alloc(0);
  const ptr = ex.alloc(vals.length * 4);
  if (ptr === 0) throw new Error('OOM: f32 alloc');
  const view = new Float32Array(ex.memory.buffer, ptr, vals.length);
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i];
    if (v === 'NaN') view[i] = NaN;
    else if (v === 'Infinity') view[i] = Infinity;
    else if (v === '-Infinity') view[i] = -Infinity;
    else view[i] = v as number;
  }
  return ptr;
}

/** Allocate a zeroed i32 output buffer. Returns ptr. */
function allocOutI32(ex: HashExports, len: number): number {
  if (len === 0) return ex.alloc(0);
  const ptr = ex.alloc(len * 4);
  if (ptr === 0) throw new Error('OOM: out i32 alloc');
  return ptr;
}

/** Allocate a zeroed i64 output buffer. Returns ptr. */
function allocOutI64(ex: HashExports, len: number): number {
  if (len === 0) return ex.alloc(0);
  const ptr = ex.alloc(len * 8);
  if (ptr === 0) throw new Error('OOM: out i64 alloc');
  return ptr;
}

/**
 * Pack validity bitmap from a fixture `vp` array (or undefined for all-valid).
 * Returns ptr; 0 means all-valid (no bitmap allocated).
 */
function packVp(ex: HashExports, vpArr: number[] | undefined, elemCount: number): number {
  if (!vpArr || vpArr.length === 0) return 0; // all-valid fast path
  const byteLen = (elemCount + 7) >> 3;
  const ptr = ex.alloc(byteLen);
  if (ptr === 0) throw new Error('OOM: vp alloc');
  const buf = new Uint8Array(ex.memory.buffer, ptr, byteLen);
  packBitmap(vpArr, buf);
  return ptr;
}

// ---------------------------------------------------------------------------
// Hash value extraction helpers
// ---------------------------------------------------------------------------

/**
 * Read `count` i64 hashes as bigint[] from the WASM buffer (copied, not a view).
 */
function readI64(ex: HashExports, ptr: number, count: number): bigint[] {
  if (count === 0) return [];
  const view = new BigInt64Array(ex.memory.buffer, ptr, count);
  return Array.from(view);
}

// ---------------------------------------------------------------------------
// Property verification helpers
// ---------------------------------------------------------------------------

/**
 * Verify the `equal_inputs_equal_hashes` property:
 * positions sharing a group in `equalGroups` must produce identical hashes;
 * positions in different groups must produce different hashes.
 */
function verifyEqualGroups(
  hashes: bigint[],
  equalGroups: number[][],
  caseName: string,
): void {
  for (const grp of equalGroups) {
    if (grp.length < 2) continue;
    const ref = hashes[grp[0]!];
    for (let k = 1; k < grp.length; k++) {
      expect(
        hashes[grp[k]!],
        `${caseName}: positions ${grp[0]} and ${grp[k]} should hash equal`,
      ).toBe(ref);
    }
  }
  for (let a = 0; a < equalGroups.length; a++) {
    for (let b = a + 1; b < equalGroups.length; b++) {
      const ha = hashes[equalGroups[a]![0]!];
      const hb = hashes[equalGroups[b]![0]!];
      expect(
        ha,
        `${caseName}: groups ${a} and ${b} should hash differently`,
      ).not.toBe(hb);
    }
  }
}

/**
 * Verify the `group_partition` property (set-of-sets, unordered).
 */
function verifyPartitions(
  outGroupIds: number[],
  expectedCount: number,
  expectedPartitions: number[][],
  caseName: string,
): void {
  const actual = new Map<number, number[]>();
  for (let i = 0; i < outGroupIds.length; i++) {
    const g = outGroupIds[i]!;
    if (!actual.has(g)) actual.set(g, []);
    actual.get(g)!.push(i);
  }

  expect(actual.size, `${caseName}: group_count`).toBe(expectedCount);

  const sortedKey = (a: number[]) => [...a].sort((x, y) => x - y).join(',');
  const actualSets = Array.from(actual.values())
    .map((a) => [...a].sort((x, y) => x - y))
    .sort((a, b) => sortedKey(a).localeCompare(sortedKey(b)));
  const expectedSets = expectedPartitions
    .map((p) => [...p].sort((x, y) => x - y))
    .sort((a, b) => sortedKey(a).localeCompare(sortedKey(b)));

  expect(actualSets, `${caseName}: partitions`).toEqual(expectedSets);
}

// ---------------------------------------------------------------------------
// hash_dt case runner
// ---------------------------------------------------------------------------

function runHashDt(ex: HashExports, c: FixtureCase): void {
  const inp = c.inputs as { data: (number | string)[]; vp?: number[] };
  const data = inp.data ?? [];
  const len = data.length;
  const vpArr = inp.vp as number[] | undefined;

  let dataPtr = 0;
  let vpPtr = 0;
  let outPtr = 0;
  try {
    const dtype = c.export.replace('hash_', '');
    switch (dtype) {
      case 'i32':
        dataPtr = allocI32(ex, data as number[]);
        break;
      case 'u32':
        dataPtr = allocU32(ex, data as number[]);
        break;
      case 'f64':
        dataPtr = allocF64(ex, data);
        break;
      case 'f32':
        dataPtr = allocF32(ex, data);
        break;
      default:
        throw new Error(`Unknown hash dtype: ${dtype}`);
    }

    vpPtr = packVp(ex, vpArr, len);
    outPtr = allocOutI64(ex, len);

    const hashFn =
      dtype === 'i32' ? ex.hash_i32
      : dtype === 'u32' ? ex.hash_u32
      : dtype === 'f64' ? ex.hash_f64
      : ex.hash_f32;
    hashFn(dataPtr, vpPtr, outPtr, len);

    const hashes = readI64(ex, outPtr, len);

    if (c.property === 'equal_inputs_equal_hashes') {
      const exp = c.expected as { equal_groups: number[][] };
      verifyEqualGroups(hashes, exp.equal_groups, c.name);
    } else {
      const exp = c.expected as { out_hash?: unknown[] };
      if (exp.out_hash !== undefined) {
        expect(hashes.length, `${c.name}: hash length`).toBe(exp.out_hash.length);
      }
    }
  } finally {
    if (dataPtr) ex.free(dataPtr);
    if (vpPtr) ex.free(vpPtr);
    if (outPtr) ex.free(outPtr);
  }
}

// ---------------------------------------------------------------------------
// hash_combine case runner
// ---------------------------------------------------------------------------

function runHashCombine(ex: HashExports, c: FixtureCase): void {
  const inp = c.inputs as { acc_hash: string[]; add_hash: string[] };
  const accStrs = inp.acc_hash ?? [];
  const addStrs = inp.add_hash ?? [];
  const len = accStrs.length;

  let acc1Ptr = 0;
  let acc2Ptr = 0;
  let addPtr = 0;
  try {
    if (c.property === 'deterministic_in_place') {
      acc1Ptr = allocI64(ex, accStrs);
      acc2Ptr = allocI64(ex, accStrs);
      addPtr = allocI64(ex, addStrs);

      ex.hash_combine(acc1Ptr, addPtr, len);
      ex.hash_combine(acc2Ptr, addPtr, len);

      const out1 = readI64(ex, acc1Ptr, len);
      const out2 = readI64(ex, acc2Ptr, len);
      expect(out1, `${c.name}: deterministic`).toEqual(out2);
    } else if (c.property === 'equal_inputs_equal_hashes') {
      acc1Ptr = allocI64(ex, accStrs);
      addPtr = allocI64(ex, addStrs);
      ex.hash_combine(acc1Ptr, addPtr, len);
      const hashes = readI64(ex, acc1Ptr, len);
      const exp = c.expected as { equal_groups: number[][] };
      verifyEqualGroups(hashes, exp.equal_groups, c.name);
    } else {
      // Empty case — just verify it doesn't crash.
      acc1Ptr = allocI64(ex, accStrs);
      addPtr = allocI64(ex, addStrs);
      ex.hash_combine(acc1Ptr, addPtr, len);
    }
  } finally {
    if (acc1Ptr) ex.free(acc1Ptr);
    if (acc2Ptr) ex.free(acc2Ptr);
    if (addPtr) ex.free(addPtr);
  }
}

// ---------------------------------------------------------------------------
// group_build case runner
// ---------------------------------------------------------------------------

function runGroupBuild(ex: HashExports, c: FixtureCase): void {
  const inp = c.inputs as { keys_hash: string[] };
  const keyStrs = inp.keys_hash ?? [];
  const len = keyStrs.length;

  let hashPtr = 0;
  let outGroupIds = 0;
  try {
    hashPtr = allocI64(ex, keyStrs);
    outGroupIds = allocOutI32(ex, len);

    const groupCount = groupBuild(ex, hashPtr, len, outGroupIds);

    const exp = c.expected as { group_count: number; partitions: number[][] };

    if (len === 0) {
      expect(groupCount, `${c.name}: empty group_count`).toBe(0);
    } else {
      const groupIds = Array.from(new Int32Array(ex.memory.buffer, outGroupIds, len));
      verifyPartitions(groupIds, exp.group_count, exp.partitions, c.name);
    }
  } finally {
    if (hashPtr) ex.free(hashPtr);
    if (outGroupIds) ex.free(outGroupIds);
  }
}

// ---------------------------------------------------------------------------
// join case runner
// ---------------------------------------------------------------------------

function runJoin(ex: HashExports, c: FixtureCase, isLeft: boolean): void {
  const inp = c.inputs as {
    lh: string[];
    l_vp?: number[];
    rh: string[];
    r_vp?: number[];
  };
  const lhStrs = inp.lh ?? [];
  const rhStrs = inp.rh ?? [];
  const lVpArr = inp.l_vp as number[] | undefined;
  const rVpArr = inp.r_vp as number[] | undefined;
  const lLen = lhStrs.length;
  const rLen = rhStrs.length;

  let lhPtr = 0;
  let rhPtr = 0;
  let lVpPtr = 0;
  let rVpPtr = 0;
  try {
    lhPtr = allocI64(ex, lhStrs);
    rhPtr = allocI64(ex, rhStrs);
    lVpPtr = packVp(ex, lVpArr, lLen);
    rVpPtr = packVp(ex, rVpArr, rLen);

    const result = isLeft
      ? joinHashLeft(ex, lhPtr, lVpPtr, lLen, rhPtr, rVpPtr, rLen)
      : joinHashInner(ex, lhPtr, lVpPtr, lLen, rhPtr, rVpPtr, rLen);

    const exp = c.expected as { pairs: [number, number][] };
    const expectedPairs = exp.pairs ?? [];

    expect(result.count, `${c.name}: pair count`).toBe(expectedPairs.length);
    for (let i = 0; i < expectedPairs.length; i++) {
      const [el, er] = expectedPairs[i]!;
      expect(result.lIdx[i], `${c.name}: pair[${i}].l`).toBe(el);
      expect(result.rIdx[i], `${c.name}: pair[${i}].r`).toBe(er);
    }
  } finally {
    if (lhPtr) ex.free(lhPtr);
    if (rhPtr) ex.free(rhPtr);
    if (lVpPtr) ex.free(lVpPtr);
    if (rVpPtr) ex.free(rVpPtr);
  }
}

// ---------------------------------------------------------------------------
// Main test loop — both builds
// ---------------------------------------------------------------------------

describe('hash kernel family — conformance', () => {
  beforeAll(async () => {
    fixture = JSON.parse(await readFile(FIXTURE_PATH, 'utf8')) as {
      cases: FixtureCase[];
    };
  });

  for (const build of BUILDS) {
    describe(`[${build.label}]`, () => {
      let ex: HashExports;

      beforeAll(async () => {
        ex = await loadAllExports(build.simd);
      });

      it('exports all required hash functions', () => {
        expect(typeof ex.hash_i32).toBe('function');
        expect(typeof ex.hash_u32).toBe('function');
        expect(typeof ex.hash_f64).toBe('function');
        expect(typeof ex.hash_f32).toBe('function');
        expect(typeof ex.hash_combine).toBe('function');
        expect(typeof ex.group_build).toBe('function');
        expect(typeof ex.join_hash_inner).toBe('function');
        expect(typeof ex.join_hash_left).toBe('function');
      });

      it('fixture cases', () => {
        for (const c of fixture.cases) {
          switch (c.export) {
            case 'hash_i32':
            case 'hash_u32':
            case 'hash_f64':
            case 'hash_f32':
              runHashDt(ex, c);
              break;
            case 'hash_combine':
              runHashCombine(ex, c);
              break;
            case 'group_build':
              runGroupBuild(ex, c);
              break;
            case 'join_hash_inner':
              runJoin(ex, c, false);
              break;
            case 'join_hash_left':
              runJoin(ex, c, true);
              break;
            default:
              throw new Error(`Unknown export in fixture: ${c.export}`);
          }
        }
      });
    });
  }
});

// ---------------------------------------------------------------------------
// Retry-path tests (force tiny htCap / outCap to exercise retry loops)
// ---------------------------------------------------------------------------

describe('hash kernel family — retry paths', () => {
  for (const build of BUILDS) {
    describe(`[${build.label}]`, () => {
      let ex: HashExports;

      beforeAll(async () => {
        ex = await loadAllExports(build.simd);
      });

      it('group_build: grows HT when htCapHint forces -1', () => {
        // 8 distinct hashes, htCapHint = 1 (way too small) → retry until large enough.
        const hashes = ['1', '2', '3', '4', '5', '6', '7', '8'];
        const len = hashes.length;

        let hashPtr = 0;
        let outGroupIds = 0;
        try {
          hashPtr = allocI64(ex, hashes);
          outGroupIds = allocOutI32(ex, len);

          const count = groupBuild(ex, hashPtr, len, outGroupIds, 1 /* tiny */);

          expect(count).toBe(8);
          const ids = Array.from(new Int32Array(ex.memory.buffer, outGroupIds, len));
          // All 8 distinct hashes → group IDs 0..7 (first-occurrence order).
          const sorted = [...ids].sort((a, b) => a - b);
          expect(sorted).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
        } finally {
          if (hashPtr) ex.free(hashPtr);
          if (outGroupIds) ex.free(outGroupIds);
        }
      });

      it('join_hash_inner: grows HT when htCapHint forces -1', () => {
        const lh = ['10', '20', '30', '10'];
        const rh = ['10', '20', '30', '40'];
        const lVp = [1, 1, 1, 1];
        const rVp = [1, 1, 1, 1];

        let lhPtr = 0;
        let rhPtr = 0;
        let lVpPtr = 0;
        let rVpPtr = 0;
        try {
          lhPtr = allocI64(ex, lh);
          rhPtr = allocI64(ex, rh);
          lVpPtr = packVp(ex, lVp, lh.length);
          rVpPtr = packVp(ex, rVp, rh.length);

          // Force htCapHint = 1 to trigger ht-grow retry.
          const result = joinHashInner(
            ex,
            lhPtr, lVpPtr, lh.length,
            rhPtr, rVpPtr, rh.length,
            1, /* htCapHint */
          );

          expect(result.count).toBe(4);
          const pairs = Array.from({ length: result.count }, (_, i) => [
            result.lIdx[i],
            result.rIdx[i],
          ]);
          expect(pairs).toEqual([[0, 0], [1, 1], [2, 2], [3, 0]]);
        } finally {
          if (lhPtr) ex.free(lhPtr);
          if (rhPtr) ex.free(rhPtr);
          if (lVpPtr) ex.free(lVpPtr);
          if (rVpPtr) ex.free(rVpPtr);
        }
      });

      it('join_hash_inner: raw kernel returns n>outCap when buffer too small', () => {
        // Directly call the raw kernel with out_cap=1 when 3 pairs exist.
        // This exercises the path that returns n > out_cap so the JS wrapper
        // can then retry with a bigger buffer.
        const lh = ['100'];
        const rh = ['100', '100', '100'];
        const lVp = [1];
        const rVp = [1, 1, 1];
        const htCap = 8;

        let lhPtr = 0;
        let rhPtr = 0;
        let lVpPtr = 0;
        let rVpPtr = 0;
        let htPtr = 0;
        let outLPtr = 0;
        let outRPtr = 0;
        try {
          lhPtr = allocI64(ex, lh);
          rhPtr = allocI64(ex, rh);
          lVpPtr = packVp(ex, lVp, 1);
          rVpPtr = packVp(ex, rVp, 3);

          htPtr = ex.alloc(htCap * 16);
          new Uint8Array(ex.memory.buffer, htPtr, htCap * 16).fill(0);
          outLPtr = ex.alloc(1 * 4); // room for only 1 pair
          outRPtr = ex.alloc(1 * 4);

          // Raw kernel: outCap=1, total=3 → should return 3.
          const rawN = ex.join_hash_inner(
            lhPtr, lVpPtr, 1,
            rhPtr, rVpPtr, 3,
            htPtr, htCap,
            outLPtr, outRPtr, 1,
          );
          expect(rawN).toBe(3);

          // The JS wrapper should transparently handle this and return all 3 pairs.
          const result = joinHashInner(ex, lhPtr, lVpPtr, 1, rhPtr, rVpPtr, 3);
          expect(result.count).toBe(3);
          expect(Array.from(result.lIdx)).toEqual([0, 0, 0]);
          expect(Array.from(result.rIdx)).toEqual([0, 1, 2]);
        } finally {
          if (lhPtr) ex.free(lhPtr);
          if (rhPtr) ex.free(rhPtr);
          if (lVpPtr) ex.free(lVpPtr);
          if (rVpPtr) ex.free(rVpPtr);
          if (htPtr) ex.free(htPtr);
          if (outLPtr) ex.free(outLPtr);
          if (outRPtr) ex.free(outRPtr);
        }
      });

      it('join_hash_left: grows HT when htCapHint forces -1', () => {
        const lh = ['10', '20', '99']; // 99 has no match on right
        const rh = ['10', '20', '30'];
        const lVp = [1, 1, 1];
        const rVp = [1, 1, 1];

        let lhPtr = 0;
        let rhPtr = 0;
        let lVpPtr = 0;
        let rVpPtr = 0;
        try {
          lhPtr = allocI64(ex, lh);
          rhPtr = allocI64(ex, rh);
          lVpPtr = packVp(ex, lVp, lh.length);
          rVpPtr = packVp(ex, rVp, rh.length);

          // Force htCapHint = 1.
          const result = joinHashLeft(
            ex,
            lhPtr, lVpPtr, lh.length,
            rhPtr, rVpPtr, rh.length,
            1, /* htCapHint */
          );

          expect(result.count).toBe(3);
          const pairs = Array.from({ length: result.count }, (_, i) => [
            result.lIdx[i],
            result.rIdx[i],
          ]);
          expect(pairs).toEqual([[0, 0], [1, 1], [2, -1]]);
        } finally {
          if (lhPtr) ex.free(lhPtr);
          if (rhPtr) ex.free(rhPtr);
          if (lVpPtr) ex.free(lVpPtr);
          if (rVpPtr) ex.free(rVpPtr);
        }
      });
    });
  }
});

// Suppress the unused import warning — detectSimd is re-exported from loader
// and useful to callers that need it alongside hash kernels.
void detectSimd;
