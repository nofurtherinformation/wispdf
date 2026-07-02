/**
 * Conformance tests for the elementwise kernel family.
 * Runs all 231 fixture cases against both scalar and SIMD builds.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '../../../wasm/dist');
const FIXTURE_PATH = join(__dir, '../../../tests/conformance/fixtures/elementwise.json');

// ── Types ─────────────────────────────────────────────────────────────────────

type FixtureValue = number | string | null;
interface FixtureInputs {
  a?: FixtureValue[];
  b?: FixtureValue[];
  a_vp?: number[];
  b_vp?: number[];
  s?: FixtureValue;
  in?: FixtureValue[];
  in_vp?: number[];
  fill?: FixtureValue;
  vp?: number[];
  mask?: number[];
}
interface FixtureExpected {
  out?: FixtureValue[];
  out_vp?: number[];
  out_mask?: number[];
  out_bool?: number[];
  out_u8?: number[];
}
interface FixtureCase {
  export: string;
  name: string;
  inputs: FixtureInputs;
  expected: FixtureExpected;
}
interface WasmExports extends Record<string, (...args: number[]) => number | void> {
  memory: WebAssembly.Memory;
  alloc: (n: number) => number;
  free: (p: number) => void;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Decode JSON float sentinels to JS values. */
function sentinel(v: FixtureValue): number {
  if (v === 'NaN') return NaN;
  if (v === 'Infinity') return Infinity;
  if (v === '-Infinity') return -Infinity;
  return v as number;
}

/** Arrow-LSB pack: per-element 0/1 array → Uint8Array bitmap. */
function packBitmap(bits: number[]): Uint8Array {
  const n = bits.length;
  const bm = new Uint8Array(Math.ceil(n / 8));
  for (let i = 0; i < n; i++) if (bits[i]) bm[i >> 3] |= 1 << (i & 7);
  return bm;
}

/** Arrow-LSB unpack: Uint8Array bitmap → per-element 0/1 array of length n. */
function unpackBitmap(bm: Uint8Array, n: number): number[] {
  return Array.from({ length: n }, (_, i) => (bm[i >> 3] >> (i & 7)) & 1);
}

/** Build a typed view + ByteOffset given dtype and length, using existing memory. */
type DType = 'f64' | 'f32' | 'i32' | 'u32' | 'u8';
const BYTE_SIZE: Record<DType, number> = { f64: 8, f32: 4, i32: 4, u32: 4, u8: 1 };

function viewOf(buffer: ArrayBuffer, dtype: DType, byteOffset: number, n: number) {
  switch (dtype) {
    case 'f64': return new Float64Array(buffer, byteOffset, n);
    case 'f32': return new Float32Array(buffer, byteOffset, n);
    case 'i32': return new Int32Array(buffer, byteOffset, n);
    case 'u32': return new Uint32Array(buffer, byteOffset, n);
    default: return new Uint8Array(buffer, byteOffset, n);
  }
}

/** Write values array into WASM buffer at ptr; returns the typed view. */
function writeData(
  buf: ArrayBuffer, ptr: number, dtype: DType, vals: FixtureValue[]
): ArrayBufferView {
  const view = viewOf(buf, dtype, ptr, vals.length);
  for (let i = 0; i < vals.length; i++) {
    (view as Float64Array)[i] = sentinel(vals[i]);
  }
  return view;
}

/** Read typed data from WASM buffer. */
function readData(buf: ArrayBuffer, dtype: DType, ptr: number, n: number): number[] {
  const view = viewOf(buf, dtype, ptr, n);
  return Array.from(view) as number[];
}

/** Load WASM module (scalar or simd). */
async function loadWasm(simd: boolean): Promise<WasmExports> {
  const fname = simd ? 'simd.wasm' : 'scalar.wasm';
  const bytes = readFileSync(join(WASM_DIR, fname));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports as unknown as WasmExports;
}

/** Determine input dtype from export name. */
function inputDtype(name: string): DType {
  // cast_X_Y → X
  const castM = name.match(/^cast_(f64|f32|i32|u32|bool)_/);
  if (castM) return castM[1] === 'bool' ? 'u8' : (castM[1] as DType);
  // fill_null_X, add_X, neg_X, gt_X_mask, gt_X_scalar_mask, etc.
  const m = name.match(/_(f64|f32|i32|u32|bool)(?:$|_)/);
  if (m) return m[1] === 'bool' ? 'u8' : (m[1] as DType);
  return 'u8';
}

/** Determine output dtype from export name. */
function outputDtype(name: string): DType {
  // cast_X_Y → Y
  const castM = name.match(/^cast_\w+_(f64|f32|i32|u32|bool)$/);
  if (castM) return castM[1] === 'bool' ? 'u8' : (castM[1] as DType);
  return inputDtype(name);
}

// ── Comparison ────────────────────────────────────────────────────────────────

function compareValue(
  got: number, exp: FixtureValue, dtype: DType, caseName: string, idx: number
): void {
  if (exp === null) return; // undefined slot — skip
  const expNum = sentinel(exp);
  if (dtype === 'f64' || dtype === 'f32') {
    // bit-exact via Object.is (distinguishes -0 from +0, NaN from NaN)
    expect(Object.is(got, expNum), `${caseName}[${idx}]: got=${got} exp=${expNum}`).toBe(true);
  } else {
    expect(got, `${caseName}[${idx}]`).toBe(expNum);
  }
}

// ── Runner ────────────────────────────────────────────────────────────────────

function runCase(mod: WasmExports, c: FixtureCase): void {
  const { inputs, expected } = c;
  const buf = mod.memory.buffer;
  const fn = mod[c.export] as (...args: number[]) => void;
  if (!fn) throw new Error(`Missing export: ${c.export}`);

  // Dispatch by shape of inputs/expected
  const name = c.export;

  // ── is_null ──
  if (name === 'is_null') {
    const vp = inputs.vp!;
    const n = vp.length;
    if (n === 0) { return; } // nothing to check
    const vpBm = packBitmap(vp);
    const vpPtr = mod.alloc(vpBm.length);
    const outPtr = mod.alloc(n);
    new Uint8Array(mod.memory.buffer, vpPtr, vpBm.length).set(vpBm);
    fn(vpPtr, outPtr, n);
    const out = Array.from(new Uint8Array(mod.memory.buffer, outPtr, n));
    mod.free(vpPtr); mod.free(outPtr);
    for (let i = 0; i < n; i++) {
      if (expected.out_bool![i] !== null)
        expect(out[i], `${c.name}[${i}]`).toBe(expected.out_bool![i]);
    }
    return;
  }

  // ── expand_mask_bool ──
  if (name === 'expand_mask_bool') {
    const mask = inputs.mask!;
    const n = mask.length;
    if (n === 0) return;
    const bm = packBitmap(mask);
    const bmPtr = mod.alloc(bm.length);
    const outPtr = mod.alloc(n);
    new Uint8Array(mod.memory.buffer, bmPtr, bm.length).set(bm);
    fn(bmPtr, outPtr, n);
    const out = Array.from(new Uint8Array(mod.memory.buffer, outPtr, n));
    mod.free(bmPtr); mod.free(outPtr);
    for (let i = 0; i < n; i++) {
      if (expected.out_u8![i] !== null)
        expect(out[i], `${c.name}[${i}]`).toBe(expected.out_u8![i]);
    }
    return;
  }

  // ── validity_and / validity_or ──
  if (name === 'validity_and' || name === 'validity_or') {
    const avp = inputs.a_vp!;
    const bvp = inputs.b_vp!;
    const n = avp.length;
    const nbytes = Math.ceil(n / 8);
    if (n === 0) return;
    const aBm = packBitmap(avp);
    const bBm = packBitmap(bvp);
    const aPtr = mod.alloc(nbytes);
    const bPtr = mod.alloc(nbytes);
    const outPtr = mod.alloc(nbytes);
    new Uint8Array(mod.memory.buffer, aPtr, nbytes).set(aBm);
    new Uint8Array(mod.memory.buffer, bPtr, nbytes).set(bBm);
    fn(aPtr, bPtr, outPtr, n);
    const outBm = new Uint8Array(mod.memory.buffer, outPtr, nbytes);
    const outVp = unpackBitmap(outBm, n);
    mod.free(aPtr); mod.free(bPtr); mod.free(outPtr);
    for (let i = 0; i < n; i++) {
      if (expected.out_vp![i] !== null)
        expect(outVp[i], `${c.name}[${i}]`).toBe(expected.out_vp![i]);
    }
    return;
  }

  // ── and_kleene / or_kleene ──
  if (name === 'and_kleene' || name === 'or_kleene') {
    const a = inputs.a!.map(Number);
    const b = inputs.b!.map(Number);
    const n = a.length;
    if (n === 0) return;
    const nbytes = Math.ceil(n / 8);
    // data is per-element u8; validity is bitmap
    const aPtr = mod.alloc(n); new Uint8Array(mod.memory.buffer, aPtr, n).set(a);
    const bPtr = mod.alloc(n); new Uint8Array(mod.memory.buffer, bPtr, n).set(b);
    const avBm = inputs.a_vp ? packBitmap(inputs.a_vp) : null;
    const bvBm = inputs.b_vp ? packBitmap(inputs.b_vp) : null;
    const avPtr = avBm ? mod.alloc(avBm.length) : 0;
    const bvPtr = bvBm ? mod.alloc(bvBm.length) : 0;
    if (avBm && avPtr) new Uint8Array(mod.memory.buffer, avPtr, avBm.length).set(avBm);
    if (bvBm && bvPtr) new Uint8Array(mod.memory.buffer, bvPtr, bvBm.length).set(bvBm);
    const outPtr = mod.alloc(n);
    const outVpPtr = mod.alloc(nbytes);
    fn(aPtr, avPtr, bPtr, bvPtr, outPtr, outVpPtr, n);
    const out = Array.from(new Uint8Array(mod.memory.buffer, outPtr, n));
    const outVp = unpackBitmap(new Uint8Array(mod.memory.buffer, outVpPtr, nbytes), n);
    mod.free(aPtr); mod.free(bPtr);
    if (avPtr) mod.free(avPtr);
    if (bvPtr) mod.free(bvPtr);
    mod.free(outPtr); mod.free(outVpPtr);
    for (let i = 0; i < n; i++) {
      if (expected.out_vp![i] !== null)
        expect(outVp[i], `${c.name} out_vp[${i}]`).toBe(expected.out_vp![i]);
      if (expected.out![i] !== null && outVp[i] === 1)
        expect(out[i], `${c.name} out[${i}]`).toBe(expected.out![i]);
    }
    return;
  }

  // ── not_bool ──
  if (name === 'not_bool') {
    const a = inputs.a!.map(Number);
    const n = a.length;
    if (n === 0) return;
    const nbytes = Math.ceil(n / 8);
    const aPtr = mod.alloc(n); new Uint8Array(mod.memory.buffer, aPtr, n).set(a);
    const avBm = inputs.a_vp ? packBitmap(inputs.a_vp) : null;
    const avPtr = avBm ? mod.alloc(avBm.length) : 0;
    if (avBm && avPtr) new Uint8Array(mod.memory.buffer, avPtr, avBm.length).set(avBm);
    const outPtr = mod.alloc(n);
    const outVpPtr = mod.alloc(nbytes);
    fn(aPtr, avPtr, outPtr, outVpPtr, n);
    const out = Array.from(new Uint8Array(mod.memory.buffer, outPtr, n));
    const outVp = unpackBitmap(new Uint8Array(mod.memory.buffer, outVpPtr, nbytes), n);
    mod.free(aPtr); if (avPtr) mod.free(avPtr); mod.free(outPtr); mod.free(outVpPtr);
    for (let i = 0; i < n; i++) {
      if (expected.out_vp![i] !== null)
        expect(outVp[i], `${c.name} out_vp[${i}]`).toBe(expected.out_vp![i]);
      if (expected.out![i] !== null && outVp[i] === 1)
        expect(out[i], `${c.name} out[${i}]`).toBe(expected.out![i]);
    }
    return;
  }

  // ── cast ops ──
  if (name.startsWith('cast_')) {
    const inp = (inputs.in ?? []) as FixtureValue[];
    const n = inp.length;
    if (n === 0) return;
    const nbytes = Math.ceil(n / 8);
    const inDtype = inputDtype(name);
    const outDtype = outputDtype(name);
    const inBs = BYTE_SIZE[inDtype];
    const outBs = BYTE_SIZE[outDtype];
    const inPtr = mod.alloc(n * inBs);
    writeData(mod.memory.buffer, inPtr, inDtype, inp);
    const invBm = inputs.in_vp ? packBitmap(inputs.in_vp) : null;
    const invPtr = invBm ? mod.alloc(invBm.length) : 0;
    if (invBm && invPtr) new Uint8Array(mod.memory.buffer, invPtr, invBm.length).set(invBm);
    const outPtr = mod.alloc(n * outBs);
    const outVpPtr = mod.alloc(nbytes);
    fn(inPtr, invPtr, outPtr, outVpPtr, n);
    const out = readData(mod.memory.buffer, outDtype, outPtr, n);
    const outVp = unpackBitmap(new Uint8Array(mod.memory.buffer, outVpPtr, nbytes), n);
    mod.free(inPtr); if (invPtr) mod.free(invPtr); mod.free(outPtr); mod.free(outVpPtr);
    for (let i = 0; i < n; i++) {
      if (expected.out_vp![i] !== null)
        expect(outVp[i], `${c.name} out_vp[${i}]`).toBe(expected.out_vp![i]);
      if (expected.out![i] !== null && outVp[i] === 1)
        compareValue(out[i], expected.out![i], outDtype, c.name, i);
    }
    return;
  }

  // ── fill_null_* ──
  if (name.startsWith('fill_null_')) {
    const inp = (inputs.in ?? []) as FixtureValue[];
    const n = inp.length;
    if (n === 0) return;
    const dtype = inputDtype(name);
    const bs = BYTE_SIZE[dtype];
    const fill = sentinel(inputs.fill!);
    const inPtr = mod.alloc(n * bs);
    writeData(mod.memory.buffer, inPtr, dtype, inp);
    const invBm = inputs.in_vp ? packBitmap(inputs.in_vp) : null;
    const invPtr = invBm ? mod.alloc(invBm.length) : 0;
    if (invBm && invPtr) new Uint8Array(mod.memory.buffer, invPtr, invBm.length).set(invBm);
    const outPtr = mod.alloc(n * bs);
    fn(inPtr, invPtr, fill, outPtr, n);
    const out = readData(mod.memory.buffer, dtype, outPtr, n);
    mod.free(inPtr); if (invPtr) mod.free(invPtr); mod.free(outPtr);
    for (let i = 0; i < n; i++) {
      if (expected.out![i] !== null)
        compareValue(out[i], expected.out![i], dtype, c.name, i);
    }
    return;
  }

  // ── Comparison ops with out_mask ──
  const isCmpScalar = /^(gt|ge|lt|le|eq|ne)_(f64|f32|i32|u32)_scalar_mask$/.test(name);
  const isCmpVec    = /^(gt|ge|lt|le|eq|ne)_(f64|f32|i32|u32)_mask$/.test(name);
  if (isCmpVec || isCmpScalar) {
    const a = (inputs.a ?? []) as FixtureValue[];
    const n = a.length;
    const nbytes = Math.ceil(n / 8);
    const dtype = inputDtype(name);
    const bs = BYTE_SIZE[dtype];
    const aPtr = n ? mod.alloc(n * bs) : 0;
    if (n) writeData(mod.memory.buffer, aPtr, dtype, a);
    const outPtr = nbytes ? mod.alloc(nbytes) : mod.alloc(1);
    if (isCmpVec) {
      const b = (inputs.b ?? []) as FixtureValue[];
      const bPtr = n ? mod.alloc(n * bs) : 0;
      if (n) writeData(mod.memory.buffer, bPtr, dtype, b);
      fn(aPtr, bPtr, outPtr, n);
      if (n) mod.free(bPtr);
    } else {
      const s = sentinel(inputs.s!);
      fn(aPtr, s, outPtr, n);
    }
    const outMask = unpackBitmap(new Uint8Array(mod.memory.buffer, outPtr, nbytes), n);
    if (n) mod.free(aPtr);
    mod.free(outPtr);
    for (let i = 0; i < n; i++) {
      if (expected.out_mask![i] !== null)
        expect(outMask[i], `${c.name}[${i}]`).toBe(expected.out_mask![i]);
    }
    return;
  }

  // ── Unary neg_* ──
  if (name.startsWith('neg_')) {
    const a = (inputs.a ?? []) as FixtureValue[];
    const n = a.length;
    if (n === 0) return;
    const dtype = inputDtype(name);
    const bs = BYTE_SIZE[dtype];
    const aPtr = mod.alloc(n * bs);
    writeData(mod.memory.buffer, aPtr, dtype, a);
    const outPtr = mod.alloc(n * bs);
    fn(aPtr, outPtr, n);
    const out = readData(mod.memory.buffer, dtype, outPtr, n);
    mod.free(aPtr); mod.free(outPtr);
    for (let i = 0; i < n; i++) {
      if (expected.out![i] !== null)
        compareValue(out[i], expected.out![i], dtype, c.name, i);
    }
    return;
  }

  // ── Scalar binary ops: add_f64_scalar, etc. ──
  const isScalarOp = /^(add|sub|mul|div|mod)_(f64|f32|i32|u32)_scalar$/.test(name);
  if (isScalarOp) {
    const a = (inputs.a ?? []) as FixtureValue[];
    const n = a.length;
    if (n === 0) return;
    const dtype = inputDtype(name);
    const bs = BYTE_SIZE[dtype];
    const s = sentinel(inputs.s!);
    const aPtr = mod.alloc(n * bs);
    writeData(mod.memory.buffer, aPtr, dtype, a);
    const outPtr = mod.alloc(n * bs);
    fn(aPtr, s, outPtr, n);
    const out = readData(mod.memory.buffer, dtype, outPtr, n);
    mod.free(aPtr); mod.free(outPtr);
    for (let i = 0; i < n; i++) {
      if (expected.out![i] !== null)
        compareValue(out[i], expected.out![i], dtype, c.name, i);
    }
    return;
  }

  // ── Binary vector ops: add_f64, sub_f32, etc. ──
  {
    const a = (inputs.a ?? []) as FixtureValue[];
    const b = (inputs.b ?? []) as FixtureValue[];
    const n = a.length;
    if (n === 0) return;
    const dtype = inputDtype(name);
    const bs = BYTE_SIZE[dtype];
    const aPtr = mod.alloc(n * bs);
    const bPtr = mod.alloc(n * bs);
    writeData(mod.memory.buffer, aPtr, dtype, a);
    writeData(mod.memory.buffer, bPtr, dtype, b);
    const outPtr = mod.alloc(n * bs);
    fn(aPtr, bPtr, outPtr, n);
    const out = readData(mod.memory.buffer, dtype, outPtr, n);
    mod.free(aPtr); mod.free(bPtr); mod.free(outPtr);
    for (let i = 0; i < n; i++) {
      if (expected.out![i] !== null)
        compareValue(out[i], expected.out![i], dtype, c.name, i);
    }
  }
}

// ── Test fixture loading ───────────────────────────────────────────────────────

const { cases } = JSON.parse(readFileSync(FIXTURE_PATH, 'utf8')) as { cases: FixtureCase[] };

// ── Test suite ────────────────────────────────────────────────────────────────

const BUILDS: ReadonlyArray<{ label: string; simd: boolean }> = [
  { label: 'scalar', simd: false },
  { label: 'simd',   simd: true  },
];

for (const { label, simd } of BUILDS) {
  describe(`elementwise conformance (${label})`, () => {
    let mod: WasmExports;

    it('loads the wasm module', async () => {
      mod = await loadWasm(simd);
      expect(mod.memory).toBeInstanceOf(WebAssembly.Memory);
      expect(typeof mod.alloc).toBe('function');
    });

    for (const c of cases) {
      it(c.name, () => {
        runCase(mod, c);
      });
    }
  });
}
