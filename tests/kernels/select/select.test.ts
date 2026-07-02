/**
 * Conformance tests for the Phase-2 selection kernel family (Agent C).
 *
 * Loads the orchestrator-owned fixture file and runs every case against both
 * scalar.wasm and simd.wasm, verifying byte-identical results per the runner
 * protocol in tests/conformance/README.md.
 *
 * Bitmap packing: Arrow LSB — bit i lives in byte[i>>3] at position (i&7).
 * Float specials in JSON:  "NaN", "Infinity", "-Infinity".
 * Null expected slots (JSON null): implementation-defined, skipped in compare.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import fixtureData from '../../conformance/fixtures/selection.json' assert { type: 'json' };

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIST = join(__dir, '..', '..', '..', 'wasm', 'dist');

// ── WASM exports interface ────────────────────────────────────────────────────

interface SelectWasm {
  memory: WebAssembly.Memory;
  alloc(size: number): number;
  free(ptr: number): void;
  realloc(ptr: number, newSize: number): number;
  mem_generation(): number;
  // filter
  filter_f64(data: number, mask: number, out: number, len: number): number;
  filter_f32(data: number, mask: number, out: number, len: number): number;
  filter_i32(data: number, mask: number, out: number, len: number): number;
  filter_u32(data: number, mask: number, out: number, len: number): number;
  filter_u8(data: number, mask: number, out: number, len: number): number;
  // filter_indices
  filter_indices(mask: number, out_idx: number, len: number): number;
  // gather
  gather_f64(data: number, idx: number, idx_len: number, out: number): void;
  gather_f32(data: number, idx: number, idx_len: number, out: number): void;
  gather_i32(data: number, idx: number, idx_len: number, out: number): void;
  gather_u32(data: number, idx: number, idx_len: number, out: number): void;
  gather_u8(data: number, idx: number, idx_len: number, out: number): void;
  gather_validity(vp: number, idx: number, idx_len: number, out_vp: number): void;
  // argsort — ABI v1.2: (data, vp, inout_perm, len, desc, scratch_ptr)
  argsort_f64(data: number, vp: number, inout_perm: number, len: number, desc: number, scratch_ptr: number): void;
  argsort_f32(data: number, vp: number, inout_perm: number, len: number, desc: number, scratch_ptr: number): void;
  argsort_i32(data: number, vp: number, inout_perm: number, len: number, desc: number, scratch_ptr: number): void;
  argsort_u32(data: number, vp: number, inout_perm: number, len: number, desc: number, scratch_ptr: number): void;
  // topk
  topk_f64(data: number, vp: number, k: number, out_idx: number, len: number, largest: number): number;
  topk_f32(data: number, vp: number, k: number, out_idx: number, len: number, largest: number): number;
  topk_i32(data: number, vp: number, k: number, out_idx: number, len: number, largest: number): number;
  topk_u32(data: number, vp: number, k: number, out_idx: number, len: number, largest: number): number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Load a fresh wasm module instance for a test build.
 *
 * We instantiate the binary directly rather than going through `loadWasmModule`
 * because the loader only surfaces the 5 Phase-1 memory exports. We need ALL
 * exports (the selection kernels) available on the same object.
 */
async function loadMod(simd: boolean): Promise<SelectWasm> {
  const name = simd ? 'simd.wasm' : 'scalar.wasm';
  const bytes = await readFile(join(WASM_DIST, name));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports as unknown as SelectWasm;
}

/** Parse a float special from JSON ("NaN", "Infinity", "-Infinity") or return as-is. */
function parseNum(v: number | string): number {
  if (v === 'NaN') return NaN;
  if (v === 'Infinity') return Infinity;
  if (v === '-Infinity') return -Infinity;
  return v as number;
}

/** Allocate `n * bytesPerElem` bytes; return pointer (or 0 for n=0). */
function allocN(mod: SelectWasm, n: number, bytesPerElem: number): number {
  if (n === 0) return mod.alloc(0); // alloc(0) returns a valid non-null ptr
  const p = mod.alloc(n * bytesPerElem);
  if (p === 0) throw new Error('OOM in test alloc');
  return p;
}

/** Pack a numeric array into wasm linear memory; return pointer. */
function packNums(
  mod: SelectWasm,
  values: (number | string)[],
  TypedArrayCtor: new (buf: ArrayBufferLike, off: number, len: number) => { [i: number]: number },
  bytesPerElem: number,
): number {
  const n = values.length;
  const ptr = allocN(mod, n, bytesPerElem);
  if (n === 0) return ptr;
  const view = new TypedArrayCtor(mod.memory.buffer, ptr, n);
  for (let i = 0; i < n; i++) {
    view[i] = parseNum(values[i] as number | string);
  }
  return ptr;
}

/** Pack a 0/1 array into an Arrow-LSB bitmap; return pointer. */
function packBitmap(mod: SelectWasm, bits: number[], len: number): number {
  const byteCount = Math.ceil(len / 8);
  if (byteCount === 0) return mod.alloc(0);
  const ptr = mod.alloc(byteCount);
  if (ptr === 0) throw new Error('OOM in packBitmap');
  const u8 = new Uint8Array(mod.memory.buffer, ptr, byteCount);
  u8.fill(0);
  for (let i = 0; i < len; i++) {
    if (bits[i]) u8[i >> 3] |= 1 << (i & 7);
  }
  return ptr;
}

/** Read an Arrow-LSB bitmap back as a 0/1 array. */
function readBitmap(mod: SelectWasm, ptr: number, len: number): number[] {
  if (len === 0) return [];
  const byteCount = Math.ceil(len / 8);
  const u8 = new Uint8Array(mod.memory.buffer, ptr, byteCount);
  const out: number[] = [];
  for (let i = 0; i < len; i++) {
    out.push((u8[i >> 3] >> (i & 7)) & 1);
  }
  return out;
}

/** Dtype byte width. */
function bytesFor(dt: string): number {
  if (dt === 'f64') return 8;
  if (dt === 'f32' || dt === 'i32' || dt === 'u32') return 4;
  return 1; // u8/bool
}

/** TypedArray constructor for a dtype. */
function arrayCtorFor(dt: string): new (buf: ArrayBufferLike, off: number, len: number) => ArrayLike<number> & { [i: number]: number } {
  if (dt === 'f64') return Float64Array;
  if (dt === 'f32') return Float32Array;
  if (dt === 'i32') return Int32Array;
  if (dt === 'u32') return Uint32Array;
  return Uint8Array;
}

/** Read a numeric array from wasm memory. */
function readNums(mod: SelectWasm, ptr: number, n: number, dt: string): number[] {
  if (n === 0) return [];
  const Ctor = arrayCtorFor(dt);
  const view = new Ctor(mod.memory.buffer, ptr, n) as Float64Array;
  return Array.from(view);
}

/**
 * Compare two float values treating NaN===NaN (bit-pattern compare).
 * A JSON null expected value means "implementation-defined; skip".
 */
function floatsEq(actual: number, expected: number | string | null): boolean {
  if (expected === null) return true;
  const exp = parseNum(expected as number | string);
  return Object.is(actual, exp);
}

/** Extract the dtype from an export name, e.g. "filter_f64" → "f64". */
function dtypeOf(exportName: string): string {
  const m = exportName.match(/_([a-z0-9]+)$/);
  return m ? m[1]! : 'f64';
}

// ── Fixture types ─────────────────────────────────────────────────────────────

type FixtureValue = number | string;

interface FilterInputs {
  data: FixtureValue[];
  mask: number[];
}
interface FilterExpected {
  out: FixtureValue[];
  count: number;
}

interface FilterIndicesInputs {
  mask: number[];
}
interface FilterIndicesExpected {
  out_idx: number[];
  count: number;
}

interface GatherInputs {
  data: FixtureValue[];
  idx: number[];
  idx_len: number;
}
interface GatherExpected {
  out: FixtureValue[];
}

interface GatherValidityInputs {
  vp: number[];
  idx: number[];
  idx_len: number;
}
interface GatherValidityExpected {
  out_vp: number[];
}

interface ArgsortInputs {
  data: FixtureValue[];
  vp?: number[];
  inout_perm: number[];
  desc: number;
}
interface ArgsortExpected {
  inout_perm: number[];
}

interface TopkInputs {
  data: FixtureValue[];
  vp?: number[];
  k: number;
  largest: number;
}
interface TopkExpected {
  out_idx: number[];
  count: number;
}

interface FixtureCase {
  export: string;
  name: string;
  note?: string;
  inputs: FilterInputs | FilterIndicesInputs | GatherInputs | GatherValidityInputs | ArgsortInputs | TopkInputs;
  expected: FilterExpected | FilterIndicesExpected | GatherExpected | GatherValidityExpected | ArgsortExpected | TopkExpected;
}

interface FixtureFile {
  family: string;
  cases: FixtureCase[];
}

// ── Runner ────────────────────────────────────────────────────────────────────

/**
 * Run one fixture case against a loaded wasm module.
 * Allocates all buffers, calls the kernel, reads output, frees buffers.
 */
function runCase(mod: SelectWasm, c: FixtureCase): void {
  const exp = c.export;
  const ptrs: number[] = [];

  const a = (n: number) => { ptrs.push(n); return n; };

  try {
    if (exp.startsWith('filter_') && exp !== 'filter_indices') {
      // filter_dt: (data, mask, out, len) -> i32 (count)
      const inp = c.inputs as FilterInputs;
      const xp = c.expected as FilterExpected;
      const dt = dtypeOf(exp);
      const bpe = bytesFor(dt);
      const Ctor = arrayCtorFor(dt);
      const len = inp.data.length;

      const dataPtr = a(packNums(mod, inp.data, Ctor, bpe));
      const maskPtr = a(packBitmap(mod, inp.mask, len));
      const outPtr  = a(allocN(mod, len, bpe)); // worst-case all selected

      // Call the kernel
      const fn = (mod as unknown as Record<string, Function>)[exp] as (...args: number[]) => number;
      const count = fn(dataPtr, maskPtr, outPtr, len);

      // Verify count
      expect(count, `${c.name}: count`).toBe(xp.count);

      // Verify output values
      const outData = readNums(mod, outPtr, count, dt);
      for (let i = 0; i < xp.out.length; i++) {
        expect(
          floatsEq(outData[i]!, xp.out[i]!),
          `${c.name}: out[${i}] actual=${outData[i]} expected=${xp.out[i]}`,
        ).toBe(true);
      }

    } else if (exp === 'filter_indices') {
      const inp = c.inputs as FilterIndicesInputs;
      const xp = c.expected as FilterIndicesExpected;
      const len = inp.mask.length;

      const maskPtr = a(packBitmap(mod, inp.mask, len));
      const outPtr  = a(allocN(mod, len, 4)); // worst-case all selected

      const count = mod.filter_indices(maskPtr, outPtr, len);

      expect(count, `${c.name}: count`).toBe(xp.count);

      const outIdx = new Int32Array(mod.memory.buffer, outPtr, count);
      for (let i = 0; i < xp.out_idx.length; i++) {
        expect(outIdx[i], `${c.name}: out_idx[${i}]`).toBe(xp.out_idx[i]);
      }

    } else if (exp.startsWith('gather_') && exp !== 'gather_validity') {
      // gather_dt: (data, idx, idx_len, out) -> ()
      const inp = c.inputs as GatherInputs;
      const xp = c.expected as GatherExpected;
      const dt = dtypeOf(exp);
      const bpe = bytesFor(dt);
      const Ctor = arrayCtorFor(dt);
      const dataLen = inp.data.length;
      const idxLen  = inp.idx_len;

      const dataPtr = a(packNums(mod, inp.data, Ctor, bpe));
      const idxPtr  = a(packNums(mod, inp.idx, Int32Array, 4));
      const outPtr  = a(allocN(mod, idxLen, bpe));

      const fn = (mod as unknown as Record<string, Function>)[exp] as (...args: number[]) => void;
      fn(dataPtr, idxPtr, idxLen, outPtr);

      const outData = readNums(mod, outPtr, idxLen, dt);
      for (let i = 0; i < xp.out.length; i++) {
        expect(
          floatsEq(outData[i]!, xp.out[i]!),
          `${c.name}: out[${i}] actual=${outData[i]} expected=${xp.out[i]}`,
        ).toBe(true);
      }
      void dataLen;

    } else if (exp === 'gather_validity') {
      const inp = c.inputs as GatherValidityInputs;
      const xp = c.expected as GatherValidityExpected;
      const idxLen = inp.idx_len;
      const srcLen = inp.vp.length;

      const vpPtr   = a(packBitmap(mod, inp.vp, srcLen));
      const idxPtr  = a(packNums(mod, inp.idx, Int32Array, 4));
      const outVpPtr = a(allocN(mod, Math.max(1, Math.ceil(idxLen / 8)), 1));
      // zero the output bitmap before the call
      new Uint8Array(mod.memory.buffer, outVpPtr, Math.ceil(idxLen / 8) || 1).fill(0);

      mod.gather_validity(vpPtr, idxPtr, idxLen, outVpPtr);

      const outBits = readBitmap(mod, outVpPtr, idxLen);
      expect(outBits, `${c.name}: out_vp`).toEqual(xp.out_vp);

    } else if (exp.startsWith('argsort_')) {
      // argsort_dt: (data, vp, inout_perm, len, desc, scratch_ptr) -> ()  [ABI v1.2]
      const inp = c.inputs as ArgsortInputs;
      const xp = c.expected as ArgsortExpected;
      const dt = dtypeOf(exp);
      const bpe = bytesFor(dt);
      const Ctor = arrayCtorFor(dt);
      const len = inp.data.length;

      const dataPtr    = a(packNums(mod, inp.data, Ctor, bpe));
      const vpPtr      = inp.vp ? a(packBitmap(mod, inp.vp, len)) : 0;
      const permPtr    = a(packNums(mod, inp.inout_perm, Int32Array, 4));
      // Allocate scratch (i32[len]) required by ABI v1.2
      const scratchPtr = a(allocN(mod, Math.max(len, 1), 4));

      const fn = (mod as unknown as Record<string, Function>)[exp] as (...args: number[]) => void;
      fn(dataPtr, vpPtr, permPtr, len, inp.desc, scratchPtr);

      const perm = new Int32Array(mod.memory.buffer, permPtr, len);
      for (let i = 0; i < xp.inout_perm.length; i++) {
        expect(perm[i], `${c.name}: inout_perm[${i}]`).toBe(xp.inout_perm[i]);
      }

    } else if (exp.startsWith('topk_')) {
      // topk_dt: (data, vp, k, out_idx, len, largest) -> i32 (count)
      const inp = c.inputs as TopkInputs;
      const xp = c.expected as TopkExpected;
      const dt = dtypeOf(exp);
      const bpe = bytesFor(dt);
      const Ctor = arrayCtorFor(dt);
      const len = inp.data.length;
      const k   = inp.k;

      const dataPtr  = a(packNums(mod, inp.data, Ctor, bpe));
      const vpPtr    = inp.vp ? a(packBitmap(mod, inp.vp, len)) : 0;
      const outIdxPtr = a(allocN(mod, Math.max(1, k), 4));

      const fn = (mod as unknown as Record<string, Function>)[exp] as (...args: number[]) => number;
      const count = fn(dataPtr, vpPtr, k, outIdxPtr, len, inp.largest);

      expect(count, `${c.name}: count`).toBe(xp.count);

      const outIdx = new Int32Array(mod.memory.buffer, outIdxPtr, count);
      for (let i = 0; i < xp.out_idx.length; i++) {
        expect(outIdx[i], `${c.name}: out_idx[${i}]`).toBe(xp.out_idx[i]);
      }

    } else {
      throw new Error(`Unknown export: ${exp}`);
    }
  } finally {
    for (const p of ptrs) {
      if (p !== 0) mod.free(p);
    }
  }
}

// ── Test suites ───────────────────────────────────────────────────────────────

const fixture = fixtureData as unknown as FixtureFile;
const BUILDS = [
  { label: 'scalar', simd: false },
  { label: 'simd',   simd: true  },
] as const;

for (const { label, simd } of BUILDS) {
  describe(`selection kernels (${label})`, () => {
    let mod: SelectWasm;

    beforeEach(async () => {
      mod = await loadMod(simd);
    });

    for (const c of fixture.cases) {
      it(`${c.export} — ${c.name}`, () => {
        runCase(mod, c);
      });
    }
  });
}
