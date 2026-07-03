/**
 * Conformance tests for i64 kernels (v2.3) — all four families.
 *
 * Loads tests/conformance/fixtures/i64.json and runs every non-frame_error
 * case against both scalar.wasm and simd.wasm.
 *
 * Fixture encoding:
 *   - i64 column data: decimal strings → BigInt64Array
 *   - Scalar bigint args (s, fill): decimal strings → BigInt
 *   - Reduction results returning i64: decimal string in expected.result
 *   - f64 returns: JSON number or "NaN"
 *   - Validity bitmaps (vp/*_vp): 0/1 integer arrays (Arrow LSB)
 *   - Hash cases: property "equal_inputs_equal_hashes" (no exact value check)
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dir = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = join(__dir, '../conformance/fixtures/i64.json');
const WASM_DIR = join(__dir, '../../wasm/dist');

// ── Types ─────────────────────────────────────────────────────────────────────

interface WasmMod {
  memory: WebAssembly.Memory;
  alloc(n: number): number;
  free(ptr: number): void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: (...args: any[]) => any;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type FixtureCase = Record<string, any>;

// ── Wasm loader ───────────────────────────────────────────────────────────────

async function loadWasm(name: 'scalar' | 'simd'): Promise<WasmMod> {
  const bytes = readFileSync(join(WASM_DIR, `${name}.wasm`));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports as unknown as WasmMod;
}

// ── Memory helpers ────────────────────────────────────────────────────────────

function alloc(w: WasmMod, bytes: number): number {
  if (bytes === 0) return w.alloc(0);
  const p = w.alloc(bytes);
  if (p === 0) throw new Error('OOM');
  return p;
}

/** Pack i64 decimal strings into wasm BigInt64Array. */
function allocI64(w: WasmMod, strs: (string | null)[]): number {
  const n = strs.length;
  const ptr = alloc(w, n * 8);
  if (n > 0) {
    const v = new BigInt64Array(w.memory.buffer, ptr, n);
    for (let i = 0; i < n; i++) v[i] = BigInt((strs[i] as string) ?? '0');
  }
  return ptr;
}

function readI64(w: WasmMod, ptr: number, n: number): bigint[] {
  const v = new BigInt64Array(w.memory.buffer, ptr, n);
  return Array.from(v);
}

function allocF64(w: WasmMod, vals: (number | string)[]): number {
  const n = vals.length;
  const ptr = alloc(w, n * 8);
  if (n > 0) {
    const v = new Float64Array(w.memory.buffer, ptr, n);
    for (let i = 0; i < n; i++) v[i] = parseSentinel(vals[i]!);
  }
  return ptr;
}

function readF64(w: WasmMod, ptr: number, n: number): number[] {
  return Array.from(new Float64Array(w.memory.buffer, ptr, n));
}

function allocF32(w: WasmMod, vals: (number | string)[]): number {
  const n = vals.length;
  const ptr = alloc(w, n * 4);
  if (n > 0) {
    const v = new Float32Array(w.memory.buffer, ptr, n);
    for (let i = 0; i < n; i++) v[i] = parseSentinel(vals[i]!);
  }
  return ptr;
}

function readF32(w: WasmMod, ptr: number, n: number): number[] {
  return Array.from(new Float32Array(w.memory.buffer, ptr, n));
}

function allocI32(w: WasmMod, vals: number[]): number {
  const n = vals.length;
  const ptr = alloc(w, n * 4);
  if (n > 0) new Int32Array(w.memory.buffer, ptr, n).set(vals);
  return ptr;
}

function readI32(w: WasmMod, ptr: number, n: number): number[] {
  return Array.from(new Int32Array(w.memory.buffer, ptr, n));
}

function allocU32(w: WasmMod, vals: number[]): number {
  const n = vals.length;
  const ptr = alloc(w, n * 4);
  if (n > 0) new Uint32Array(w.memory.buffer, ptr, n).set(vals);
  return ptr;
}

function readU32(w: WasmMod, ptr: number, n: number): number[] {
  return Array.from(new Uint32Array(w.memory.buffer, ptr, n));
}

function allocU8(w: WasmMod, vals: number[]): number {
  const n = vals.length;
  const ptr = alloc(w, n);
  if (n > 0) new Uint8Array(w.memory.buffer, ptr, n).set(vals);
  return ptr;
}

function readU8(w: WasmMod, ptr: number, n: number): number[] {
  return Array.from(new Uint8Array(w.memory.buffer, ptr, n));
}

/** Pack Arrow-LSB bitmap; returns ptr (or 0 if bits is undefined). */
function allocBitmap(w: WasmMod, bits: number[] | undefined, n: number): number {
  if (bits === undefined) return 0;
  const bytes = Math.ceil(n / 8);
  const ptr = alloc(w, Math.max(bytes, 1));
  const buf = new Uint8Array(w.memory.buffer, ptr, Math.max(bytes, 1));
  buf.fill(0);
  for (let i = 0; i < bits.length; i++) {
    if (bits[i]) buf[i >> 3]! |= 1 << (i & 7);
  }
  return ptr;
}

function unpackBitmap(buf: Uint8Array, n: number): number[] {
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = (buf[i >> 3]! >> (i & 7)) & 1;
  return out;
}

function parseSentinel(v: number | string): number {
  if (v === 'NaN') return NaN;
  if (v === 'Infinity') return Infinity;
  if (v === '-Infinity') return -Infinity;
  return v as number;
}

/** Bit-exact f64 equality (handles NaN). */
function f64Eq(a: number, b: number): boolean {
  const buf = new ArrayBuffer(8);
  const view = new DataView(buf);
  view.setFloat64(0, a, true);
  const aBits = view.getBigUint64(0, true);
  view.setFloat64(0, b, true);
  return aBits === view.getBigUint64(0, true);
}

/** Bit-exact f32 equality (handles NaN). */
function f32Eq(a: number, b: number): boolean {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  view.setFloat32(0, a, true);
  const aBits = view.getUint32(0, true);
  view.setFloat32(0, b, true);
  return aBits === view.getUint32(0, true);
}

// ── Case runners ──────────────────────────────────────────────────────────────

const BINARY_VEC = new Set(['add_i64', 'sub_i64', 'mul_i64', 'div_i64', 'mod_i64']);
const SCALAR_VEC = new Set(['add_i64_scalar', 'sub_i64_scalar', 'mul_i64_scalar', 'div_i64_scalar', 'mod_i64_scalar']);
const MASK_VEC   = new Set(['gt_i64_mask', 'ge_i64_mask', 'lt_i64_mask', 'le_i64_mask', 'eq_i64_mask', 'ne_i64_mask']);
const MASK_SCL   = new Set(['gt_i64_scalar_mask', 'ge_i64_scalar_mask', 'lt_i64_scalar_mask', 'le_i64_scalar_mask', 'eq_i64_scalar_mask', 'ne_i64_scalar_mask']);
const BIGINT_RED = new Set(['sum_i64_null', 'min_i64_null', 'max_i64_null']);
const F64_RED    = new Set(['mean_i64_null', 'std_i64_null', 'var_i64_null']);

function runCase(w: WasmMod, c: FixtureCase): void {
  if (c.layer === 'frame_error') return;

  const fn = c.export as string;
  const inp = c.inputs;
  const exp = c.expected;

  // ── Binary vector arithmetic: add_i64, sub_i64, mul_i64, div_i64, mod_i64 ──
  if (BINARY_VEC.has(fn)) {
    const a = inp.a as string[];
    const b = inp.b as string[];
    const n = a.length;
    const aPtr = allocI64(w, a);
    const bPtr = allocI64(w, b);
    const outPtr = alloc(w, n * 8);
    try {
      (w[fn] as Function)(aPtr, bPtr, outPtr, n);
      const expOut = exp.out as (string | null)[];
      const got = new BigInt64Array(w.memory.buffer, outPtr, n);
      for (let i = 0; i < n; i++) {
        if (expOut[i] === null) continue;
        expect(got[i]).toBe(BigInt(expOut[i]!));
      }
    } finally { w.free(aPtr); w.free(bPtr); w.free(outPtr); }
    return;
  }

  // ── Scalar arithmetic: add_i64_scalar, etc. ──────────────────────────────
  if (SCALAR_VEC.has(fn)) {
    const a = inp.a as string[];
    const s = BigInt(inp.s as string);
    const n = a.length;
    const aPtr = allocI64(w, a);
    const outPtr = alloc(w, n * 8);
    try {
      (w[fn] as Function)(aPtr, s, outPtr, n);
      const expOut = exp.out as (string | null)[];
      const got = new BigInt64Array(w.memory.buffer, outPtr, n);
      for (let i = 0; i < n; i++) {
        if (expOut[i] === null) continue;
        expect(got[i]).toBe(BigInt(expOut[i]!));
      }
    } finally { w.free(aPtr); w.free(outPtr); }
    return;
  }

  // ── neg_i64 ──────────────────────────────────────────────────────────────
  if (fn === 'neg_i64') {
    const a = inp.a as string[];
    const n = a.length;
    const aPtr = allocI64(w, a);
    const outPtr = alloc(w, n * 8);
    try {
      (w['neg_i64'] as Function)(aPtr, outPtr, n);
      const expOut = exp.out as (string | null)[];
      const got = new BigInt64Array(w.memory.buffer, outPtr, n);
      for (let i = 0; i < n; i++) {
        if (expOut[i] === null) continue;
        expect(got[i]).toBe(BigInt(expOut[i]!));
      }
    } finally { w.free(aPtr); w.free(outPtr); }
    return;
  }

  // ── Comparison masks ──────────────────────────────────────────────────────
  if (MASK_VEC.has(fn)) {
    const a = inp.a as string[];
    const b = inp.b as string[];
    const n = a.length;
    const maskBytes = Math.ceil(n / 8);
    const aPtr = allocI64(w, a);
    const bPtr = allocI64(w, b);
    const outPtr = alloc(w, Math.max(maskBytes, 1));
    try {
      (w[fn] as Function)(aPtr, bPtr, outPtr, n);
      const expMask = exp.out_mask as number[];
      const gotBuf = new Uint8Array(w.memory.buffer, outPtr, Math.max(maskBytes, 1));
      const got = unpackBitmap(gotBuf, n);
      expect(got).toEqual(expMask);
    } finally { w.free(aPtr); w.free(bPtr); w.free(outPtr); }
    return;
  }

  if (MASK_SCL.has(fn)) {
    const a = inp.a as string[];
    const s = BigInt(inp.s as string);
    const n = a.length;
    const maskBytes = Math.ceil(n / 8);
    const aPtr = allocI64(w, a);
    const outPtr = alloc(w, Math.max(maskBytes, 1));
    try {
      (w[fn] as Function)(aPtr, s, outPtr, n);
      const expMask = exp.out_mask as number[];
      const gotBuf = new Uint8Array(w.memory.buffer, outPtr, Math.max(maskBytes, 1));
      const got = unpackBitmap(gotBuf, n);
      expect(got).toEqual(expMask);
    } finally { w.free(aPtr); w.free(outPtr); }
    return;
  }

  // ── cast_* ────────────────────────────────────────────────────────────────
  if (fn.startsWith('cast_')) {
    runCastCase(w, c);
    return;
  }

  // ── fill_null_i64 ─────────────────────────────────────────────────────────
  if (fn === 'fill_null_i64') {
    const inStrs = inp.in as string[];
    const vp = inp.vp as number[] | undefined;
    const fill = inp.fill !== undefined ? BigInt(inp.fill as string) : 0n;
    const n = inStrs.length;
    const inPtr  = allocI64(w, inStrs);
    const vpPtr  = allocBitmap(w, vp, n);
    const outPtr = alloc(w, n * 8);
    try {
      (w['fill_null_i64'] as Function)(inPtr, vpPtr, fill, outPtr, n);
      const expOut = exp.out as (string | null)[];
      const got = new BigInt64Array(w.memory.buffer, outPtr, n);
      for (let i = 0; i < n; i++) {
        if (expOut[i] === null) continue;
        expect(got[i]).toBe(BigInt(expOut[i]!));
      }
    } finally {
      w.free(inPtr);
      if (vpPtr) w.free(vpPtr);
      w.free(outPtr);
    }
    return;
  }

  // ── Reductions ────────────────────────────────────────────────────────────
  if (BIGINT_RED.has(fn)) {
    const data = inp.data as string[];
    const vp   = inp.vp as number[] | undefined;
    const n    = data.length;
    const dPtr = allocI64(w, data);
    const vpPtr = allocBitmap(w, vp, n);
    try {
      const result = (w[fn] as Function)(dPtr, vpPtr, n) as bigint;
      expect(result).toBe(BigInt(exp.result as string));
    } finally { w.free(dPtr); if (vpPtr) w.free(vpPtr); }
    return;
  }

  if (F64_RED.has(fn)) {
    const data = inp.data as string[];
    const vp   = inp.vp as number[] | undefined;
    const n    = data.length;
    const dPtr = allocI64(w, data);
    const vpPtr = allocBitmap(w, vp, n);
    try {
      const result = (w[fn] as Function)(dPtr, vpPtr, n) as number;
      const expVal = parseSentinel(exp.result as number | string);
      expect(f64Eq(result, expVal)).toBe(true);
    } finally { w.free(dPtr); if (vpPtr) w.free(vpPtr); }
    return;
  }

  if (fn === 'nunique_i64_null') {
    const data = inp.data as string[];
    const vp   = inp.vp as number[] | undefined;
    const n    = data.length;
    const dPtr = allocI64(w, data);
    const vpPtr = allocBitmap(w, vp, n);
    try {
      const result = (w[fn] as Function)(dPtr, vpPtr, n) as number;
      expect(result).toBe(exp.result as number);
    } finally { w.free(dPtr); if (vpPtr) w.free(vpPtr); }
    return;
  }

  if (fn === 'first_i64_null' || fn === 'last_i64_null') {
    const data = inp.data as string[];
    const vp   = inp.vp as number[] | undefined;
    const n    = data.length;
    const dPtr = allocI64(w, data);
    const vpPtr = allocBitmap(w, vp, n);
    const ovPtr = alloc(w, 4);  // i32 out_valid
    try {
      const result = (w[fn] as Function)(dPtr, vpPtr, n, ovPtr) as bigint;
      const outValid = new Int32Array(w.memory.buffer, ovPtr, 1)[0]!;
      expect(outValid).toBe(exp.out_valid as number);
      if (exp.result !== undefined) {
        expect(result).toBe(BigInt(exp.result as string));
      }
    } finally {
      w.free(dPtr); if (vpPtr) w.free(vpPtr); w.free(ovPtr);
    }
    return;
  }

  // ── filter_i64 ────────────────────────────────────────────────────────────
  if (fn === 'filter_i64') {
    const data = inp.data as string[];
    const mask = inp.mask as number[];
    const n    = data.length;
    const dPtr   = allocI64(w, data);
    const mPtr   = allocBitmap(w, mask, n);
    const outPtr = alloc(w, n * 8 || 8);
    try {
      const count = (w['filter_i64'] as Function)(dPtr, mPtr, outPtr, n) as number;
      expect(count).toBe(exp.count as number);
      const expOut = exp.out as string[];
      const got = new BigInt64Array(w.memory.buffer, outPtr, count);
      for (let i = 0; i < expOut.length; i++) {
        expect(got[i]).toBe(BigInt(expOut[i]!));
      }
    } finally { w.free(dPtr); if (mPtr) w.free(mPtr); w.free(outPtr); }
    return;
  }

  // ── gather_i64 ────────────────────────────────────────────────────────────
  if (fn === 'gather_i64') {
    const data   = inp.data as string[];
    const idx    = inp.idx as number[];
    const idxLen = inp.idx_len as number;
    const dPtr   = allocI64(w, data);
    const iPtr   = allocI32(w, idx);
    const outPtr = alloc(w, idxLen * 8 || 8);
    try {
      (w['gather_i64'] as Function)(dPtr, iPtr, idxLen, outPtr);
      const expOut = exp.out as string[];
      const got = new BigInt64Array(w.memory.buffer, outPtr, idxLen);
      for (let i = 0; i < expOut.length; i++) {
        expect(got[i]).toBe(BigInt(expOut[i]!));
      }
    } finally { w.free(dPtr); w.free(iPtr); w.free(outPtr); }
    return;
  }

  // ── argsort_i64 ───────────────────────────────────────────────────────────
  if (fn === 'argsort_i64') {
    const data = inp.data as string[];
    const vp   = inp.vp as number[] | undefined;
    const perm = inp.inout_perm as number[];
    const desc = inp.desc as number;
    const n    = perm.length;
    const dPtr  = allocI64(w, data);
    const vpPtr = allocBitmap(w, vp, n);
    const pPtr  = allocI32(w, perm);
    const sPtr  = alloc(w, n * 4 || 4);
    try {
      (w['argsort_i64'] as Function)(dPtr, vpPtr, pPtr, n, desc, sPtr);
      const expPerm = exp.out_perm as number[];
      const got = new Int32Array(w.memory.buffer, pPtr, n);
      expect(Array.from(got)).toEqual(expPerm);
    } finally {
      w.free(dPtr); if (vpPtr) w.free(vpPtr); w.free(pPtr); w.free(sPtr);
    }
    return;
  }

  // ── topk_i64 ─────────────────────────────────────────────────────────────
  if (fn === 'topk_i64') {
    const data    = inp.data as string[];
    const vp      = inp.vp as number[] | undefined;
    const k       = inp.k as number;
    const len     = inp.len !== undefined ? (inp.len as number) : data.length;
    const largest = inp.largest as number;
    const dPtr    = allocI64(w, data);
    const vpPtr   = allocBitmap(w, vp, len);
    const oPtr    = alloc(w, k * 4 || 4);
    try {
      const count = (w['topk_i64'] as Function)(dPtr, vpPtr, k, oPtr, len, largest) as number;
      expect(count).toBe(exp.count as number);
      const expIdx = exp.out_idx as number[];
      const got = new Int32Array(w.memory.buffer, oPtr, count);
      expect(Array.from(got)).toEqual(expIdx);
    } finally {
      w.free(dPtr); if (vpPtr) w.free(vpPtr); w.free(oPtr);
    }
    return;
  }

  // ── hash_i64 ─────────────────────────────────────────────────────────────
  if (fn === 'hash_i64') {
    const data = inp.data as string[];
    const vp   = inp.vp as number[];
    const n    = data.length;
    const dPtr   = allocI64(w, data);
    const vpPtr  = allocBitmap(w, vp, n);
    const outPtr = alloc(w, n * 8 || 8);
    try {
      (w['hash_i64'] as Function)(dPtr, vpPtr, outPtr, n);
      // Property: equal (value, validity) groups → equal hashes
      const hashes = new BigInt64Array(w.memory.buffer, outPtr, n);
      const groups = new Map<string, number[]>();
      for (let i = 0; i < n; i++) {
        const key = vp[i] ? `v:${data[i]}` : 'null';
        (groups.get(key) ?? (groups.set(key, []), groups.get(key)!)).push(i);
      }
      for (const indices of groups.values()) {
        const h0 = hashes[indices[0]!]!;
        for (const idx of indices) expect(hashes[idx]).toBe(h0);
      }
    } finally {
      w.free(dPtr); if (vpPtr) w.free(vpPtr); w.free(outPtr);
    }
    return;
  }

  throw new Error(`Unknown export: ${fn}`);
}

// ── Cast case handler ─────────────────────────────────────────────────────────

function runCastCase(w: WasmMod, c: FixtureCase): void {
  const fn  = c.export as string;
  const inp = c.inputs;
  const exp = c.expected;

  // Determine src/dst types from export name: cast_<src>_<dst>
  const [, src, dst] = fn.split('_') as [string, string, string];

  // Input data
  const inVals = inp.in as (number | string)[];
  const inVp   = inp.in_vp as number[] | undefined;
  const n      = inVals.length;
  const vpPtr  = allocBitmap(w, inVp, n);

  // Allocate input buffer
  let inPtr: number;
  if (src === 'f64')  inPtr = allocF64(w, inVals);
  else if (src === 'f32') inPtr = allocF32(w, inVals as (number | string)[]);
  else if (src === 'i32') inPtr = allocI32(w, inVals as number[]);
  else if (src === 'u32') inPtr = allocU32(w, inVals as number[]);
  else if (src === 'bool') inPtr = allocU8(w, inVals as number[]);
  else /* src === 'i64' */ inPtr = allocI64(w, inVals as string[]);

  // Output element size
  const dstBytes = dst === 'f64' ? 8 : dst === 'f32' ? 4 : dst === 'i64' ? 8 : dst === 'i32' ? 4 : dst === 'u32' ? 4 : 1;
  const outPtr   = alloc(w, n * dstBytes || 8);
  const ovpBytes = Math.ceil(n / 8) || 1;
  const ovpPtr   = alloc(w, ovpBytes);

  try {
    (w[fn] as Function)(inPtr, vpPtr, outPtr, ovpPtr, n);

    // Compare out_vp
    if (exp.out_vp !== undefined) {
      const expVp = exp.out_vp as number[];
      const gotVp = unpackBitmap(new Uint8Array(w.memory.buffer, ovpPtr, ovpBytes), n);
      expect(gotVp).toEqual(expVp);
    }

    // Compare out values
    const expOut = exp.out as (string | number | null)[];
    if (dst === 'i64') {
      const got = new BigInt64Array(w.memory.buffer, outPtr, n);
      for (let i = 0; i < n; i++) {
        if (expOut[i] === null) continue;
        expect(got[i]).toBe(BigInt(expOut[i] as string));
      }
    } else if (dst === 'f64') {
      const got = new Float64Array(w.memory.buffer, outPtr, n);
      for (let i = 0; i < n; i++) {
        if (expOut[i] === null) continue;
        expect(f64Eq(got[i]!, parseSentinel(expOut[i] as number | string))).toBe(true);
      }
    } else if (dst === 'f32') {
      const got = new Float32Array(w.memory.buffer, outPtr, n);
      for (let i = 0; i < n; i++) {
        if (expOut[i] === null) continue;
        expect(f32Eq(got[i]!, parseSentinel(expOut[i] as number | string))).toBe(true);
      }
    } else if (dst === 'i32') {
      const got = new Int32Array(w.memory.buffer, outPtr, n);
      for (let i = 0; i < n; i++) {
        if (expOut[i] === null) continue;
        expect(got[i]).toBe(expOut[i] as number);
      }
    } else if (dst === 'u32') {
      const got = new Uint32Array(w.memory.buffer, outPtr, n);
      for (let i = 0; i < n; i++) {
        if (expOut[i] === null) continue;
        expect(got[i]).toBe(expOut[i] as number);
      }
    } else /* bool */ {
      const got = new Uint8Array(w.memory.buffer, outPtr, n);
      for (let i = 0; i < n; i++) {
        if (expOut[i] === null) continue;
        expect(got[i]).toBe(expOut[i] as number);
      }
    }
  } finally {
    w.free(inPtr);
    if (vpPtr) w.free(vpPtr);
    w.free(outPtr);
    w.free(ovpPtr);
  }
}

// ── Test setup ────────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let fixture: { cases: FixtureCase[] };
let scalar: WasmMod;
let simd: WasmMod;

beforeAll(async () => {
  fixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf-8')) as typeof fixture;
  [scalar, simd] = await Promise.all([loadWasm('scalar'), loadWasm('simd')]);
});

describe('i64 kernels — scalar build', () => {
  it('all fixture cases pass', () => {
    for (const c of fixture.cases) runCase(scalar, c);
  });
});

describe('i64 kernels — simd build', () => {
  it('all fixture cases pass', () => {
    for (const c of fixture.cases) runCase(simd, c);
  });
});
