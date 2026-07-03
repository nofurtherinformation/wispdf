/**
 * bench/kernels/i64.mjs — i64 kernel benchmarks (v2.3)
 *
 * Gate: WASM add_i64 ≥ 1.5× faster than JS BigInt64Array loop @1M elements.
 * Also benchmarks hash_i64 vs JS BigInt splitmix64 loop.
 *
 * Usage: node bench/kernels/i64.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { Bench } from 'tinybench';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '../../wasm/dist');

const N = 1_000_000;
const BENCH_TIME = 1000;
const WARMUP = 3;

async function loadWasm(name) {
  const bytes = readFileSync(join(WASM_DIR, `${name}.wasm`));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

// ── JS BigInt splitmix64 baseline ─────────────────────────────────────────────

function splitmix64(x) {
  x += 0x9e3779b97f4a7c15n;
  x = BigInt.asIntN(64, ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n));
  x = BigInt.asIntN(64, ((x ^ (x >> 27n)) * 0x94d049bb133111ebn));
  return BigInt.asIntN(64, x ^ (x >> 31n));
}

// ── Setup ─────────────────────────────────────────────────────────────────────

const simd   = await loadWasm('simd');
const scalar = await loadWasm('scalar');

// Wasm buffers
const aPtr   = simd.alloc(N * 8);
const bPtr   = simd.alloc(N * 8);
const outPtr = simd.alloc(N * 8);
const vpPtr  = 0; // all-valid

const aView  = new BigInt64Array(simd.memory.buffer, aPtr, N);
const bView  = new BigInt64Array(simd.memory.buffer, bPtr, N);

// JS-side arrays (same data)
const jsA    = new BigInt64Array(N);
const jsB    = new BigInt64Array(N);
const jsOut  = new BigInt64Array(N);

// Fill with deterministic data
for (let i = 0; i < N; i++) {
  const v = BigInt(i + 1) * 1000003n;
  aView[i]  = v;
  bView[i]  = v ^ 0xdeadbeefdeadbeefn;
  jsA[i]    = v;
  jsB[i]    = v ^ 0xdeadbeefdeadbeefn;
}

// Shared scalar-build buffers (same pointers; scalar build has same memory layout)
const scalarAPtr   = scalar.alloc(N * 8);
const scalarBPtr   = scalar.alloc(N * 8);
const scalarOutPtr = scalar.alloc(N * 8);
new BigInt64Array(scalar.memory.buffer, scalarAPtr, N).set(jsA);
new BigInt64Array(scalar.memory.buffer, scalarBPtr, N).set(jsB);

// ── Benchmarks ────────────────────────────────────────────────────────────────

const bench = new Bench({ time: BENCH_TIME, warmupIterations: WARMUP });

// Gated: WASM add_i64 vs JS BigInt64Array loop
bench.add('wasm:simd add_i64 @1M',   () => simd.add_i64(aPtr, bPtr, outPtr, N));
bench.add('wasm:scl  add_i64 @1M',   () => scalar.add_i64(scalarAPtr, scalarBPtr, scalarOutPtr, N));
bench.add('js   BigInt64Array add @1M', () => {
  for (let i = 0; i < N; i++) jsOut[i] = jsA[i] + jsB[i];
});

// hash_i64 vs JS BigInt splitmix64 loop
bench.add('wasm:simd hash_i64 @1M',  () => simd.hash_i64(aPtr, vpPtr, outPtr, N));
bench.add('wasm:scl  hash_i64 @1M',  () => scalar.hash_i64(scalarAPtr, vpPtr, scalarOutPtr, N));
bench.add('js   BigInt splitmix64 @1M', () => {
  for (let i = 0; i < N; i++) jsOut[i] = splitmix64(jsA[i]);
});

// mul_i64 (no SIMD path in either build, still faster than BigInt)
bench.add('wasm:simd mul_i64 @1M',   () => simd.mul_i64(aPtr, bPtr, outPtr, N));
bench.add('js   BigInt64Array mul @1M', () => {
  for (let i = 0; i < N; i++) jsOut[i] = jsA[i] * jsB[i];
});

await bench.run();

console.log('\n=== i64 kernel benchmark results ===');
console.table(
  bench.tasks.map(t => ({
    name:       t.name,
    'ops/s':    (t.result?.hz ?? 0).toFixed(1),
    'avg (ms)': (t.result?.mean ? (t.result.mean * 1000).toFixed(3) : '-'),
  }))
);

// ── Gate check ────────────────────────────────────────────────────────────────

function getHz(name) {
  return bench.tasks.find(t => t.name === name)?.result?.hz ?? 0;
}

const wasmAddHz = getHz('wasm:simd add_i64 @1M');
const jsAddHz   = getHz('js   BigInt64Array add @1M');
const addRatio  = wasmAddHz / jsAddHz;

const wasmHashHz = getHz('wasm:simd hash_i64 @1M');
const jsHashHz   = getHz('js   BigInt splitmix64 @1M');
const hashRatio  = wasmHashHz / jsHashHz;

console.log(`\nadd_i64:  wasm/js ratio = ${addRatio.toFixed(2)}×  (gate ≥ 1.5×) ${addRatio >= 1.5 ? '✓ PASS' : '✗ FAIL'}`);
console.log(`hash_i64: wasm/js ratio = ${hashRatio.toFixed(2)}×  (informational)`);

if (addRatio < 1.5) {
  console.error('\nERROR: add_i64 wasm/js ratio below 1.5× gate');
  process.exit(1);
}
