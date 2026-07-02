/**
 * Benchmark: reduce kernel family vs TypedArray JS loops at 1M rows.
 *
 * Standalone node script — `node bench/kernels/reduce.mjs`
 *
 * Uses tinybench + mulberry32 seeded data (bench/datasets.mjs conventions).
 * Loads simd.wasm (SIMD build) and compares to equivalent TypedArray loops.
 *
 * Gate: ≥1.5× for kernels that have a direct JS analog.
 *
 * KNOWN CEILING — nunique:
 *   The v1 kernel uses an O(n²) in-place scan. For 1M rows with high
 *   cardinality data the JS Set approach is O(n), so the wasm kernel will
 *   be dramatically slower. This is flagged as a v1 limitation; v2 needs
 *   a caller-provided hash-set scratch or sort-based dedup. The bench
 *   reports absolute numbers for nunique and marks it CEILING.
 */

import { Bench } from 'tinybench';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { mulberry32 } from '../datasets.mjs';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '../../wasm/dist');

const N = 1_000_000;
const NULL_RATE = 0.05;
const BENCH_TIME_MS = 2000;
const BENCH_WARMUP = 3;

// ---------------------------------------------------------------------------
// WASM loader
// ---------------------------------------------------------------------------

async function loadWasm(name) {
  const buf = await readFile(join(WASM_DIR, `${name}.wasm`));
  const { instance } = await WebAssembly.instantiate(buf, {});
  return instance.exports;
}

// ---------------------------------------------------------------------------
// Seeded data generation
// ---------------------------------------------------------------------------

function makeData(n, seed = 0xdeadbeef) {
  const rng = mulberry32(seed);
  const f64 = new Float64Array(n);
  const f32 = new Float32Array(n);
  const i32 = new Int32Array(n);
  const u32 = new Uint32Array(n);
  const valid = new Uint8Array(n); // 1 = valid

  for (let i = 0; i < n; i++) {
    const isValid = rng() >= NULL_RATE;
    valid[i] = isValid ? 1 : 0;
    const v = rng() * 1e6;
    f64[i] = isValid ? v : 0.0;
    f32[i] = isValid ? v : 0.0;
    i32[i] = isValid ? (v | 0) : 0;
    u32[i] = isValid ? (v >>> 0) : 0;
  }
  return { f64, f32, i32, u32, valid, n };
}

// ---------------------------------------------------------------------------
// Bitmap packing (Arrow LSB)
// ---------------------------------------------------------------------------

function packBitmap(valid, n) {
  const bytes = new Uint8Array(Math.ceil(n / 8));
  for (let i = 0; i < n; i++) {
    if (valid[i]) bytes[i >> 3] |= 1 << (i & 7);
  }
  return bytes;
}

// ---------------------------------------------------------------------------
// WASM buffer setup
// ---------------------------------------------------------------------------

function setupWasmBuffers(wasm, data) {
  const n = data.n;
  const bitmapBytes = packBitmap(data.valid, n);

  // Allocate all pointers first (each alloc may trigger memory.grow which detaches views).
  const allocF64 = wasm.alloc(n * 8);
  const allocF32 = wasm.alloc(n * 4);
  const allocI32 = wasm.alloc(n * 4);
  const allocU32 = wasm.alloc(n * 4);
  const allocVp  = wasm.alloc(bitmapBytes.length);

  // Now get a fresh buffer reference (all allocations done, no more grows).
  const mem = wasm.memory.buffer;

  new Float64Array(mem, allocF64, n).set(data.f64);
  new Float32Array(mem, allocF32, n).set(data.f32);
  new Int32Array(mem, allocI32, n).set(data.i32);
  new Uint32Array(mem, allocU32, n).set(data.u32);
  new Uint8Array(mem, allocVp, bitmapBytes.length).set(bitmapBytes);

  return { allocF64, allocF32, allocI32, allocU32, allocVp };
}

// ---------------------------------------------------------------------------
// TypedArray JS analog loops
// ---------------------------------------------------------------------------

function jsSum(arr, valid) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) { if (valid[i]) s += arr[i]; }
  return s;
}

function jsMean(arr, valid) {
  let s = 0, c = 0;
  for (let i = 0; i < arr.length; i++) { if (valid[i]) { s += arr[i]; c++; } }
  return c ? s / c : NaN;
}

function jsMin(arr, valid) {
  let r = Infinity, f = false;
  for (let i = 0; i < arr.length; i++) {
    if (valid[i] && !isNaN(arr[i]) && arr[i] < r) { r = arr[i]; f = true; }
  }
  return f ? r : NaN;
}

function jsMax(arr, valid) {
  let r = -Infinity, f = false;
  for (let i = 0; i < arr.length; i++) {
    if (valid[i] && !isNaN(arr[i]) && arr[i] > r) { r = arr[i]; f = true; }
  }
  return f ? r : NaN;
}

function jsCountNull(valid) {
  let c = 0;
  for (let i = 0; i < valid.length; i++) { if (valid[i]) c++; }
  return c;
}

function jsStd(arr, valid) {
  let s = 0, c = 0;
  for (let i = 0; i < arr.length; i++) { if (valid[i]) { s += arr[i]; c++; } }
  if (c < 2) return NaN;
  const m = s / c;
  let ss = 0;
  for (let i = 0; i < arr.length; i++) { if (valid[i]) { const d = arr[i] - m; ss += d * d; } }
  return Math.sqrt(ss / (c - 1));
}

// nunique via JS Set (O(n) average) — this is what our O(n²) kernel competes against
function jsNunique(arr, valid) {
  const s = new Set();
  for (let i = 0; i < arr.length; i++) {
    if (valid[i]) s.add(arr[i]);
  }
  return s.size;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log(`\n=== reduce kernel bench: SIMD build vs TypedArray JS, N=${N.toLocaleString()} ===\n`);

  const N_SMALL = 2000; // fits within nunique O(n²) scratch at reasonable speed

  const data = makeData(N);
  const dataSmall = makeData(N_SMALL, 0xc0ffee);
  const wasm = await loadWasm('simd');
  const bufs = setupWasmBuffers(wasm, data);
  const { allocF64, allocF32, allocI32, allocU32, allocVp } = bufs;
  const bufsSmall = setupWasmBuffers(wasm, dataSmall);
  const { allocF64: allocF64Small, allocVp: allocVpSmall } = bufsSmall;

  // Shorthands
  const exp = wasm;
  const n = N;
  // N_SMALL already declared above

  // Kernels grouped with their JS analogs
  const ops = [
    {
      name: 'sum_f64_null',
      hasJsAnalog: true,
      wasm: () => exp.sum_f64_null(allocF64, allocVp, n),
      js:   () => jsSum(data.f64, data.valid),
    },
    {
      name: 'sum_f32_null',
      hasJsAnalog: true,
      wasm: () => exp.sum_f32_null(allocF32, allocVp, n),
      js:   () => jsSum(data.f32, data.valid),
    },
    {
      name: 'sum_i32_null',
      hasJsAnalog: true,
      wasm: () => exp.sum_i32_null(allocI32, allocVp, n),
      js:   () => jsSum(data.i32, data.valid),
    },
    {
      name: 'mean_f64_null',
      hasJsAnalog: true,
      wasm: () => exp.mean_f64_null(allocF64, allocVp, n),
      js:   () => jsMean(data.f64, data.valid),
    },
    {
      name: 'min_f64_null',
      hasJsAnalog: true,
      wasm: () => exp.min_f64_null(allocF64, allocVp, n),
      js:   () => jsMin(data.f64, data.valid),
    },
    {
      name: 'max_f64_null',
      hasJsAnalog: true,
      wasm: () => exp.max_f64_null(allocF64, allocVp, n),
      js:   () => jsMax(data.f64, data.valid),
    },
    {
      name: 'count_null',
      hasJsAnalog: true,
      wasm: () => exp.count_null(allocVp, n),
      js:   () => jsCountNull(data.valid),
    },
    {
      name: 'std_f64_null',
      hasJsAnalog: true,
      wasm: () => exp.std_f64_null(allocF64, allocVp, n),
      js:   () => jsStd(data.f64, data.valid),
    },
    // ponytail: nunique_f64_null wasm is O(n²) at N>512 distinct values; 1M rows
    // would run for hours. Bench at N=2000 (fits in 512-elem scratch) for reference
    // only — not gated. v2 fix: caller-provided hash-set scratch.
    {
      name: 'nunique_f64_null (N=2000)',
      hasJsAnalog: false,
      wasm: () => exp.nunique_f64_null(allocF64Small, allocVpSmall, N_SMALL),
      js:   () => jsNunique(dataSmall.f64, dataSmall.valid),
      ceiling: true,
    },
  ];

  const results = [];

  for (const op of ops) {
    const bench = new Bench({ time: BENCH_TIME_MS, warmupIterations: BENCH_WARMUP });
    bench.add('wasm', op.wasm);
    bench.add('js', op.js);
    await bench.run();

    const wasmResult = bench.tasks.find(t => t.name === 'wasm').result;
    const jsResult   = bench.tasks.find(t => t.name === 'js').result;

    const wasmMs = wasmResult.mean;
    const jsMs   = jsResult.mean;
    const ratio  = jsMs / wasmMs;
    const gate   = op.hasJsAnalog && !op.ceiling ? (ratio >= 1.5 ? 'PASS' : 'FAIL') : 'N/A';
    const note   = op.ceiling ? ' [CEILING: O(n²) kernel vs O(n) JS Set — v1 limitation]' : '';

    results.push({ name: op.name, wasmMs, jsMs, ratio, gate, note });

    const ratioStr = op.hasJsAnalog ? ` ratio wasm/js = ${ratio.toFixed(2)}x` : '';
    console.log(
      `${op.name.padEnd(22)} wasm=${wasmMs.toFixed(3)}ms  js=${jsMs.toFixed(3)}ms${ratioStr}  [${gate}]${note}`,
    );
  }

  console.log('\n--- Summary ---');
  const gated = results.filter(r => r.gate !== 'N/A');
  const passed = gated.filter(r => r.gate === 'PASS');
  const failed = gated.filter(r => r.gate === 'FAIL');
  const ceilings = results.filter(r => r.note.includes('CEILING'));

  console.log(`${passed.length}/${gated.length} gated kernels meet ≥1.5× gate`);
  if (failed.length > 0) {
    console.log('\nFAILED gate:');
    for (const r of failed) {
      console.log(`  ${r.name}: ${r.ratio.toFixed(2)}x (need 1.5x)`);
    }
  }
  if (ceilings.length > 0) {
    console.log('\nCEILING (not gated):');
    for (const r of ceilings) {
      console.log(`  ${r.name}: wasm=${r.wasmMs.toFixed(1)}ms js=${r.jsMs.toFixed(1)}ms — v2 fix required (caller hash-set scratch)`);
    }
  }

  // Clean up wasm memory
  for (const ptr of [allocF64, allocF32, allocI32, allocU32, allocVp,
                     allocF64Small, allocVpSmall]) {
    wasm.free(ptr);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
