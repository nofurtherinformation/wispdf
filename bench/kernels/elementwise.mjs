/**
 * bench/kernels/elementwise.mjs
 * Benchmark: SIMD WASM elementwise kernels vs TypedArray JS loops.
 *
 * Usage:
 *   node bench/kernels/elementwise.mjs
 *
 * Gate: ≥1.5× speedup vs TypedArray for kernels that have a JS analog.
 * Kernels with no JS analog (cast, fill_null, kleene, etc.) are measured but
 * only reported as absolute throughput.
 *
 * Runs inside Docker or directly on host with Node ≥ 18.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { Bench } from 'tinybench';
import { mulberry32 } from '../datasets.mjs';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '../../wasm/dist');

const N = 1_000_000;   // 1M elements
const BENCH_TIME = 1000;  // ms per benchmark
const WARMUP = 3;

// ── WASM loader ───────────────────────────────────────────────────────────────

async function loadWasm(name) {
  const bytes = readFileSync(join(WASM_DIR, `${name}.wasm`));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

// ── Seeded dataset ────────────────────────────────────────────────────────────

function makeDataset(mod) {
  const rng = mulberry32(0xdeadbeef);
  const mem = mod.memory;

  const N8  = N;         // u8 bytes
  const N4  = N * 4;     // i32/u32/f32 bytes
  const N8b = N * 8;     // f64 bytes
  const bmBytes = Math.ceil(N / 8);  // validity bitmap bytes

  // Allocate WASM buffers
  const aF64 = mod.alloc(N8b);
  const bF64 = mod.alloc(N8b);
  const aF32 = mod.alloc(N4);
  const bF32 = mod.alloc(N4);
  const aI32 = mod.alloc(N4);
  const bI32 = mod.alloc(N4);
  const aU32 = mod.alloc(N4);
  const bU32 = mod.alloc(N4);
  const outF64 = mod.alloc(N8b);
  const outF32 = mod.alloc(N4);
  const outI32 = mod.alloc(N4);
  const outU32 = mod.alloc(N4);
  const outBm  = mod.alloc(bmBytes + 1);
  const validBm = mod.alloc(bmBytes);
  const boolA  = mod.alloc(N8);
  const boolB  = mod.alloc(N8);
  const outBool = mod.alloc(N8);
  const outVp   = mod.alloc(bmBytes + 1);

  // Fill with seeded data
  const af64v = new Float64Array(mem.buffer, aF64, N);
  const bf64v = new Float64Array(mem.buffer, bF64, N);
  const af32v = new Float32Array(mem.buffer, aF32, N);
  const bf32v = new Float32Array(mem.buffer, bF32, N);
  const ai32v = new Int32Array(mem.buffer, aI32, N);
  const bi32v = new Int32Array(mem.buffer, bI32, N);
  const au32v = new Uint32Array(mem.buffer, aU32, N);
  const bu32v = new Uint32Array(mem.buffer, bU32, N);
  const boolAv = new Uint8Array(mem.buffer, boolA, N);
  const boolBv = new Uint8Array(mem.buffer, boolB, N);
  const validBmV = new Uint8Array(mem.buffer, validBm, bmBytes);

  for (let i = 0; i < N; i++) {
    const v = rng() * 1000 + 1;
    af64v[i] = v;
    bf64v[i] = rng() * 1000 + 1;  // +1 to avoid zero divisor
    af32v[i] = Math.fround(v);
    bf32v[i] = Math.fround(rng() * 1000 + 1);
    ai32v[i] = ((rng() * 2e9) | 0);
    bi32v[i] = ((rng() * 1000) | 0) + 1;  // +1: no zero divisor
    au32v[i] = (rng() * 4e9) >>> 0;
    bu32v[i] = ((rng() * 1000) >>> 0) + 1;
    boolAv[i] = rng() > 0.5 ? 1 : 0;
    boolBv[i] = rng() > 0.5 ? 1 : 0;
  }
  // all-valid bitmap (all bits = 1)
  validBmV.fill(0xff);
  // fix last byte padding
  const tail = N % 8;
  if (tail) validBmV[bmBytes - 1] = (1 << tail) - 1;

  // JS TypedArray mirrors for JS baseline
  const jsA = af64v.slice();
  const jsB = bf64v.slice();
  const jsOut = new Float64Array(N);

  return {
    aF64, bF64, aF32, bF32, aI32, bI32, aU32, bU32,
    outF64, outF32, outI32, outU32, outBm, validBm,
    boolA, boolB, outBool, outVp,
    jsA, jsB, jsOut,
    N, bmBytes,
  };
}

// ── Benchmark runner ──────────────────────────────────────────────────────────

const results = [];

async function runBench(label, wasmFn, jsFn) {
  const bench = new Bench({ time: BENCH_TIME, warmupIterations: WARMUP });
  bench.add('wasm', wasmFn);
  if (jsFn) bench.add('js', jsFn);
  await bench.run();

  const wasmTask = bench.tasks.find((t) => t.name === 'wasm');
  const jsTask = bench.tasks.find((t) => t.name === 'js');
  const wasmMs = wasmTask?.result?.mean ?? NaN;
  const jsMs = jsTask?.result?.mean ?? NaN;
  const ratio = jsFn ? jsMs / wasmMs : null;

  results.push({ label, wasmMs, jsMs, ratio });

  const wasmOps = (1000 / wasmMs).toFixed(1);
  if (jsFn) {
    const jsOps = (1000 / jsMs).toFixed(1);
    const flag = ratio !== null && ratio >= 1.5 ? 'OK' : (ratio !== null ? 'BELOW GATE' : '');
    console.log(`  ${label.padEnd(30)} wasm=${wasmOps} ops/s  js=${jsOps} ops/s  ratio=${ratio?.toFixed(2)}x  ${flag}`);
  } else {
    console.log(`  ${label.padEnd(30)} wasm=${wasmOps} ops/s  (no JS analog)`);
  }
  return { wasmMs, jsMs, ratio };
}

// ── Main ──────────────────────────────────────────────────────────────────────

console.log(`\n=== elementwise kernel bench (N=${N.toLocaleString()}) ===\n`);

const simd = await loadWasm('simd');
const ds = makeDataset(simd);
const scalar = 3.14;

// Add basic ops vs TypedArray loop baseline
const jsAddF64 = () => { for (let i = 0; i < N; i++) ds.jsOut[i] = ds.jsA[i] + ds.jsB[i]; };
const jsMulF64 = () => { for (let i = 0; i < N; i++) ds.jsOut[i] = ds.jsA[i] * ds.jsB[i]; };
const jsSubF64 = () => { for (let i = 0; i < N; i++) ds.jsOut[i] = ds.jsA[i] - ds.jsB[i]; };
const jsDivF64 = () => { for (let i = 0; i < N; i++) ds.jsOut[i] = ds.jsA[i] / ds.jsB[i]; };
const jsGtF64  = () => { const bm = new Uint8Array(ds.bmBytes); for (let i = 0; i < N; i++) if (ds.jsA[i] > ds.jsB[i]) bm[i>>3] |= 1<<(i&7); };

console.log('--- f64 binary (SIMD WASM vs TypedArray loop) ---');
await runBench('add_f64',           () => simd.add_f64(ds.aF64, ds.bF64, ds.outF64, N),             jsAddF64);
await runBench('sub_f64',           () => simd.sub_f64(ds.aF64, ds.bF64, ds.outF64, N),             jsSubF64);
await runBench('mul_f64',           () => simd.mul_f64(ds.aF64, ds.bF64, ds.outF64, N),             jsMulF64);
await runBench('div_f64',           () => simd.div_f64(ds.aF64, ds.bF64, ds.outF64, N),             jsDivF64);

console.log('\n--- f64 comparisons (SIMD WASM vs TypedArray loop) ---');
await runBench('gt_f64_mask',       () => simd.gt_f64_mask(ds.aF64, ds.bF64, ds.outBm, N),         jsGtF64);
await runBench('gt_f64_scalar_mask',() => simd.gt_f64_scalar_mask(ds.aF64, scalar, ds.outBm, N),   null);

console.log('\n--- f32 binary ---');
await runBench('add_f32',           () => simd.add_f32(ds.aF32, ds.bF32, ds.outF32, N),             null);
await runBench('mul_f32',           () => simd.mul_f32(ds.aF32, ds.bF32, ds.outF32, N),             null);

console.log('\n--- i32/u32 binary ---');
await runBench('add_i32',           () => simd.add_i32(ds.aI32, ds.bI32, ds.outI32, N),             null);
await runBench('div_i32',           () => simd.div_i32(ds.aI32, ds.bI32, ds.outI32, N),             null);
await runBench('add_u32',           () => simd.add_u32(ds.aU32, ds.bU32, ds.outU32, N),             null);

console.log('\n--- scalar ops ---');
await runBench('add_f64_scalar',    () => simd.add_f64_scalar(ds.aF64, scalar, ds.outF64, N),       null);

console.log('\n--- null-aware ---');
await runBench('and_kleene',        () => simd.and_kleene(ds.boolA, ds.validBm, ds.boolB, ds.validBm, ds.outBool, ds.outVp, N), null);
await runBench('validity_and',      () => simd.validity_and(ds.validBm, ds.validBm, ds.outBm, N),    null);
await runBench('fill_null_f64',     () => simd.fill_null_f64(ds.aF64, ds.validBm, 0.0, ds.outF64, N), null);

console.log('\n--- casts ---');
await runBench('cast_f64_i32',      () => simd.cast_f64_i32(ds.aF64, ds.validBm, ds.outI32, ds.outVp, N), null);
await runBench('cast_i32_f64',      () => simd.cast_i32_f64(ds.aI32, ds.validBm, ds.outF64, ds.outVp, N), null);

// ── Gate check ────────────────────────────────────────────────────────────────

console.log('\n=== Gate check (≥1.5× for kernels with JS analog) ===');
let gateOk = true;
for (const r of results) {
  if (r.ratio !== null) {
    const ok = r.ratio >= 1.5;
    if (!ok) gateOk = false;
    console.log(`  ${r.label.padEnd(30)} ratio=${r.ratio.toFixed(2)}x  ${ok ? 'OK' : 'FAIL (below 1.5x)'}`);
  }
}
console.log(gateOk ? '\nAll gate checks PASSED.' : '\nSome gate checks FAILED (below 1.5x).');

// Print summary table for structured output
console.log('\n=== bench_vs_typedarray ===');
for (const r of results.filter((r) => r.ratio !== null)) {
  console.log(`${r.label}: ${r.ratio.toFixed(2)}x`);
}
