/**
 * Conformance tests for the reduce kernel family (Agent B, Phase 2).
 *
 * Runs every case in tests/conformance/fixtures/reductions.json against both
 * scalar.wasm and simd.wasm.  Runner protocol: tests/conformance/README.md.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '../../../wasm/dist');
const FIXTURE_PATH = join(__dir, '../../../tests/conformance/fixtures/reductions.json');

// ---------------------------------------------------------------------------
// WASM loader
// ---------------------------------------------------------------------------

interface WasmExports {
  memory: WebAssembly.Memory;
  alloc(size: number): number;
  free(ptr: number): void;
  [key: string]: unknown;
}

async function loadBuild(name: 'scalar' | 'simd'): Promise<WasmExports> {
  const buf = await readFile(join(WASM_DIR, `${name}.wasm`));
  const { instance } = await WebAssembly.instantiate(buf, {});
  return instance.exports as unknown as WasmExports;
}

// ---------------------------------------------------------------------------
// Bitmap packing (Arrow LSB)
// ---------------------------------------------------------------------------

function packBitmap(bits: number[], len: number): Uint8Array {
  const bytes = new Uint8Array(Math.ceil(len / 8));
  for (let i = 0; i < len; i++) {
    if (bits[i]) bytes[i >> 3] |= 1 << (i & 7);
  }
  return bytes;
}

// ---------------------------------------------------------------------------
// Special float values
// ---------------------------------------------------------------------------

function parseSpecial(v: number | string): number {
  if (typeof v === 'number') return v;
  if (v === 'NaN') return NaN;
  if (v === 'Infinity') return Infinity;
  if (v === '-Infinity') return -Infinity;
  throw new Error(`Unknown special: ${v}`);
}

/** Bit-exact f64 comparison: NaN===NaN, +0!=-0. */
function bitsEqF64(a: number, b: number): boolean {
  return Object.is(a, b);
}

/** Bit-exact f32 comparison via uint32 bits. */
function bitsEqF32(a: number, b: number): boolean {
  const fa = new Float32Array(1); fa[0] = a;
  const fb = new Float32Array(1); fb[0] = b;
  return new Uint32Array(fa.buffer)[0] === new Uint32Array(fb.buffer)[0];
}

function hexF64(v: number): string {
  const b = new ArrayBuffer(8);
  new DataView(b).setFloat64(0, v, false);
  return Array.from(new Uint8Array(b)).map(x => x.toString(16).padStart(2, '0')).join('');
}

// ---------------------------------------------------------------------------
// Dtype inference from export name
// ---------------------------------------------------------------------------

function dtypeOf(exportName: string): 'f64' | 'f32' | 'i32' | 'u32' {
  if (exportName.includes('_f64')) return 'f64';
  if (exportName.includes('_f32')) return 'f32';
  if (exportName.includes('_u32')) return 'u32';
  return 'i32';
}

/** Is the return value a f64? (covers int sums and all means/std/var) */
function returnsF64(exportName: string): boolean {
  return exportName.startsWith('sum_i32') || exportName.startsWith('sum_u32')
    || exportName.startsWith('mean_')
    || exportName.startsWith('std_')
    || exportName.startsWith('var_')
    || exportName.includes('_f64');
}

/** Is the return value a f32? (min/max/first/last of f32 columns) */
function returnsF32(exportName: string): boolean {
  return exportName.startsWith('min_f32') || exportName.startsWith('max_f32')
    || exportName.startsWith('first_f32') || exportName.startsWith('last_f32');
}

// ---------------------------------------------------------------------------
// Alloc helpers
// ---------------------------------------------------------------------------

function allocData(
  wasm: WasmExports,
  dtype: 'f64' | 'f32' | 'i32' | 'u32',
  raw: (number | string)[],
): number {
  const len = raw.length;
  const elemBytes = dtype === 'f64' ? 8 : 4;
  const ptr = wasm.alloc(Math.max(len * elemBytes, 1));
  const mem = wasm.memory.buffer;
  if (dtype === 'f64') {
    const a = new Float64Array(mem, ptr, len);
    raw.forEach((v, i) => { a[i] = parseSpecial(v); });
  } else if (dtype === 'f32') {
    const a = new Float32Array(mem, ptr, len);
    raw.forEach((v, i) => { a[i] = parseSpecial(v); });
  } else if (dtype === 'i32') {
    const a = new Int32Array(mem, ptr, len);
    raw.forEach((v, i) => { a[i] = v as number; });
  } else {
    const a = new Uint32Array(mem, ptr, len);
    raw.forEach((v, i) => { a[i] = v as number; });
  }
  return ptr;
}

function allocBitmap(wasm: WasmExports, bits: number[], len: number): number {
  const packed = packBitmap(bits, len);
  const ptr = wasm.alloc(Math.max(packed.length, 1));
  new Uint8Array(wasm.memory.buffer, ptr, packed.length).set(packed);
  return ptr;
}

// ---------------------------------------------------------------------------
// Fixture schema
// ---------------------------------------------------------------------------

interface Case {
  export: string;
  name: string;
  note?: string;
  inputs: { data?: (number | string)[]; vp?: number[] };
  expected: { result?: number | string; out_valid?: number };
}

interface Fixtures { family: string; cases: Case[] }

// ---------------------------------------------------------------------------
// Single-case runner
// ---------------------------------------------------------------------------

function runCase(wasm: WasmExports, c: Case): void {
  const dtype = dtypeOf(c.export);
  const rawData = c.inputs.data ?? [];
  const rawVp = c.inputs.vp;
  // For count_null, len comes from vp array; otherwise from data.
  const len = c.export === 'count_null' ? (rawVp?.length ?? 0) : rawData.length;

  // Allocate data
  let dataPtr = 0;
  if (c.export !== 'count_null') {
    dataPtr = allocData(wasm, dtype, rawData.length > 0 ? rawData : []);
  }

  // Allocate validity bitmap (0 = all-valid fast path per ABI §4.1)
  let vpPtr = 0;
  if (rawVp !== undefined) {
    vpPtr = allocBitmap(wasm, rawVp, len);
  }

  // Allocate out_valid for first/last
  let ovPtr = 0;
  const needsOutValid = c.export.startsWith('first_') || c.export.startsWith('last_');
  if (needsOutValid) {
    ovPtr = wasm.alloc(4);
    new Int32Array(wasm.memory.buffer, ovPtr, 1)[0] = 0;
  }

  try {
    const fn = wasm[c.export] as (...args: number[]) => number;

    // Call with correct signature
    let scalar: number;
    if (c.export === 'count_null') {
      scalar = fn(vpPtr, len);
    } else if (needsOutValid) {
      scalar = fn(dataPtr, vpPtr, len, ovPtr);
    } else {
      scalar = fn(dataPtr, vpPtr, len);
    }

    // Verify out_valid for first/last
    if (c.expected.out_valid !== undefined) {
      const ov = new Int32Array(wasm.memory.buffer, ovPtr, 1)[0];
      expect(ov, `[${c.name}] out_valid`).toBe(c.expected.out_valid);
      if (c.expected.out_valid === 0) return; // scalar is impl-defined when null
    }

    // Verify scalar result
    if (c.expected.result !== undefined) {
      const want = parseSpecial(c.expected.result);
      if (returnsF64(c.export)) {
        expect(
          bitsEqF64(scalar, want),
          `[${c.name}] got ${scalar} (${hexF64(scalar)}) want ${want} (${hexF64(want)})`,
        ).toBe(true);
      } else if (returnsF32(c.export)) {
        expect(
          bitsEqF32(scalar, want),
          `[${c.name}] f32: got ${scalar} want ${want}`,
        ).toBe(true);
      } else {
        // integer / count — u32 results come back as signed i32 from wasm; convert.
        const got = c.export.includes('_u32') ? (scalar >>> 0) : scalar;
        expect(got, `[${c.name}]`).toBe(want);
      }
    }
  } finally {
    if (dataPtr) wasm.free(dataPtr);
    if (vpPtr)   wasm.free(vpPtr);
    if (ovPtr)   wasm.free(ovPtr);
  }
}

// ---------------------------------------------------------------------------
// Test suite — loaded once, run against both builds
// ---------------------------------------------------------------------------

let fixtures: Fixtures;
let scalar: WasmExports;
let simd: WasmExports;

beforeAll(async () => {
  [fixtures, scalar, simd] = await Promise.all([
    readFile(FIXTURE_PATH, 'utf-8').then(t => JSON.parse(t) as Fixtures),
    loadBuild('scalar'),
    loadBuild('simd'),
  ]);
});

describe('reduce — scalar build', () => {
  it('all 118 fixture cases pass', () => {
    for (const c of fixtures.cases) {
      runCase(scalar, c);
    }
  });
});

describe('reduce — simd build', () => {
  it('all 118 fixture cases pass', () => {
    for (const c of fixtures.cases) {
      runCase(simd, c);
    }
  });
});
