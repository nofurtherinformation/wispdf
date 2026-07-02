/**
 * Benchmark for AssemblyScript spike kernels – ADR-007.
 *
 * Kernels benchmarked (scalar + simd builds × 1M + 10M rows):
 *   add_f64        – element-wise f64 add
 *   sum_f64_null   – null-aware f64 sum (Arrow LSB bitmap, all-valid)
 *   cmp_gt_f64_mask – compare-to-scalar → LSB bitmask
 *
 * Run inside Docker:
 *   node bench.mjs
 *
 * Writes results.json to the same directory.
 */

import { readFileSync, writeFileSync } from 'fs';
import { gzip } from 'zlib';
import { promisify } from 'util';
import { performance } from 'perf_hooks';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const gzipAsync = promisify(gzip);
const __dirname = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Load and instantiate a wasm module
// ---------------------------------------------------------------------------
async function loadWasm(name) {
  const buf = readFileSync(join(__dirname, 'build', `${name}.wasm`));
  const { instance } = await WebAssembly.instantiate(buf, {
    env: { abort: () => {} },
  });
  return instance.exports;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function fillRandom(view, lo = 0, hi = 100) {
  for (let i = 0; i < view.length; i++) {
    view[i] = lo + Math.random() * (hi - lo);
  }
}

function fillAllValid(view) {
  view.fill(0xff); // all bits = 1 → all valid
}

/** Returns elapsed seconds for one run of fn() */
function timeOnce(fn) {
  const t0 = performance.now();
  fn();
  return (performance.now() - t0) / 1000; // seconds
}

/** Warm up then collect timed runs; return { ms_mean, ops_per_sec } */
function bench(fn, rows, { warmup = 5, runs = 15 } = {}) {
  for (let i = 0; i < warmup; i++) fn();
  const times = [];
  for (let i = 0; i < runs; i++) times.push(timeOnce(fn));
  const ms_mean = (times.reduce((a, b) => a + b, 0) / runs) * 1000;
  const ops_per_sec = rows / (ms_mean / 1000);
  return { ms_mean: +ms_mean.toFixed(3), ops_per_sec: Math.round(ops_per_sec) };
}

// ---------------------------------------------------------------------------
// Gzip size helper
// ---------------------------------------------------------------------------
async function gzipSize(name) {
  const buf = readFileSync(join(__dirname, 'build', `${name}.wasm`));
  const compressed = await gzipAsync(buf);
  return compressed.length;
}

// ---------------------------------------------------------------------------
// Run one build variant (scalar or simd)
// ---------------------------------------------------------------------------
async function runVariant(buildName, rowSizes) {
  console.log(`\n--- ${buildName} ---`);
  const exports_ = await loadWasm(buildName);
  const { alloc, add_f64, sum_f64_null, cmp_gt_f64_mask, memory } = exports_;

  const results = [];

  for (const rows of rowSizes) {
    const byteLen = rows * 8;
    const bitmapBytes = Math.ceil(rows / 8);

    // Allocate wasm memory for all buffers
    const a_ptr   = alloc(byteLen);
    const b_ptr   = alloc(byteLen);
    const out_ptr = alloc(byteLen);
    const validity_ptr = alloc(bitmapBytes);
    const mask_ptr = alloc(bitmapBytes);

    // Fill from JS (views into wasm linear memory)
    const memBuf = () => memory.buffer; // call each time in case memory grew
    const a_view = () => new Float64Array(memBuf(), a_ptr, rows);
    const b_view = () => new Float64Array(memBuf(), b_ptr, rows);
    const v_view = () => new Uint8Array(memBuf(), validity_ptr, bitmapBytes);

    fillRandom(a_view());
    fillRandom(b_view());
    fillAllValid(v_view());

    const scalar_val = 50.0; // median of [0,100] → ~50% pass filter

    console.log(`  [${buildName}] ${(rows / 1e6).toFixed(0)}M rows`);

    // --- add_f64 ---
    const add_stats = bench(
      () => add_f64(a_ptr, b_ptr, out_ptr, rows),
      rows,
    );
    console.log(`    add_f64        : ${add_stats.ops_per_sec.toLocaleString()} ops/s  (${add_stats.ms_mean} ms)`);
    results.push({ kernel: 'add_f64', build: buildName, rows, ...add_stats });

    // --- sum_f64_null ---
    const sum_stats = bench(
      () => sum_f64_null(a_ptr, validity_ptr, rows),
      rows,
    );
    console.log(`    sum_f64_null   : ${sum_stats.ops_per_sec.toLocaleString()} ops/s  (${sum_stats.ms_mean} ms)`);
    results.push({ kernel: 'sum_f64_null', build: buildName, rows, ...sum_stats });

    // --- cmp_gt_f64_mask ---
    const cmp_stats = bench(
      () => cmp_gt_f64_mask(a_ptr, scalar_val, mask_ptr, rows),
      rows,
    );
    console.log(`    cmp_gt_f64_mask: ${cmp_stats.ops_per_sec.toLocaleString()} ops/s  (${cmp_stats.ms_mean} ms)`);
    results.push({ kernel: 'cmp_gt_f64_mask', build: buildName, rows, ...cmp_stats });
  }

  return results;
}

// ---------------------------------------------------------------------------
// Verify correctness (quick sanity check on small data)
// ---------------------------------------------------------------------------
async function verifyCorrectness(buildName) {
  const exports_ = await loadWasm(buildName);
  const { alloc, add_f64, sum_f64_null, cmp_gt_f64_mask, memory } = exports_;

  const N = 16;
  const a_ptr  = alloc(N * 8);
  const b_ptr  = alloc(N * 8);
  const out_ptr = alloc(N * 8);
  const validity_ptr = alloc(Math.ceil(N / 8));
  const mask_ptr = alloc(Math.ceil(N / 8));

  const a = new Float64Array(memory.buffer, a_ptr, N);
  const b = new Float64Array(memory.buffer, b_ptr, N);
  const validity = new Uint8Array(memory.buffer, validity_ptr, Math.ceil(N / 8));

  for (let i = 0; i < N; i++) { a[i] = i + 1; b[i] = i * 2; }
  // validity: elements 0,2,4,... valid; odds null → byte 0 = 0b01010101 = 0x55
  validity[0] = 0x55;
  validity[1] = 0x55;

  add_f64(a_ptr, b_ptr, out_ptr, N);
  const out = new Float64Array(memory.buffer, out_ptr, N);
  let ok = true;
  for (let i = 0; i < N; i++) {
    const expected = (i + 1) + (i * 2);
    if (Math.abs(out[i] - expected) > 1e-12) {
      console.error(`  [${buildName}] add_f64 FAIL at ${i}: got ${out[i]} expected ${expected}`);
      ok = false;
    }
  }

  // sum: valid elements are 0,2,4,...14 (indices, values a[0..7 even])
  // a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
  // valid (bit=1) at positions 0,2,4,6,8,10,12,14 → sum = 1+3+5+7+9+11+13+15 = 64
  sum_f64_null(a_ptr, validity_ptr, N);
  const sumResult = sum_f64_null(a_ptr, validity_ptr, N);
  const expectedSum = 1 + 3 + 5 + 7 + 9 + 11 + 13 + 15;
  if (Math.abs(sumResult - expectedSum) > 1e-10) {
    console.error(`  [${buildName}] sum_f64_null FAIL: got ${sumResult} expected ${expectedSum}`);
    ok = false;
  }

  // cmp: a[i] = i+1, scalar=8; elements 8..15 should set bits 8..15
  cmp_gt_f64_mask(a_ptr, 8, mask_ptr, N);
  const mask = new Uint8Array(memory.buffer, mask_ptr, 2);
  // elements 0-7: a[0..7] = 1..8, none > 8 → byte 0 = 0x00
  // elements 8-15: a[8..15] = 9..16, all > 8 → byte 1 = 0xFF
  if (mask[0] !== 0x00 || mask[1] !== 0xFF) {
    console.error(`  [${buildName}] cmp_gt_f64_mask FAIL: got [${mask[0]}, ${mask[1]}] expected [0, 255]`);
    ok = false;
  }

  if (ok) {
    console.log(`  [${buildName}] correctness: PASS`);
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  console.log('AssemblyScript kernel spike – ADR-007 benchmark');
  console.log('================================================\n');

  // Correctness check first
  console.log('--- Correctness verification ---');
  const scalarOk = await verifyCorrectness('scalar');
  const simdOk   = await verifyCorrectness('simd');
  if (!scalarOk || !simdOk) {
    console.error('\nCorrectness check failed – aborting benchmark');
    process.exit(1);
  }

  // Benchmark
  const rowSizes = [1_000_000, 10_000_000];
  const scalarResults = await runVariant('scalar', rowSizes);
  const simdResults   = await runVariant('simd',   rowSizes);
  const allResults    = [...scalarResults, ...simdResults];

  // Gzip sizes
  const scalarGzip = await gzipSize('scalar');
  const simdGzip   = await gzipSize('simd');
  console.log(`\nGzip sizes: scalar=${scalarGzip} bytes, simd=${simdGzip} bytes`);

  // Build time from file written by build.sh
  let build_time_s = null;
  try {
    const btText = readFileSync(join(__dirname, 'build', 'build_time.txt'), 'utf8');
    const m = btText.match(/BUILD_TIME_S=([0-9.]+)/);
    if (m) build_time_s = parseFloat(m[1]);
  } catch (_) {}

  // Write results.json
  const output = {
    results: allResults,
    sizes_gzip_bytes: { scalar: scalarGzip, simd: simdGzip },
    build_time_s,
    timestamp: new Date().toISOString(),
  };
  const outPath = join(__dirname, 'results.json');
  writeFileSync(outPath, JSON.stringify(output, null, 2));
  console.log(`\nResults written to results.json`);

  // Summary table
  console.log('\n--- Summary (ops/s) ---');
  console.log('kernel            | build  | rows | ops/s          | ms_mean');
  console.log('------------------|--------|------|----------------|--------');
  for (const r of allResults) {
    const rowStr = r.rows >= 1e6 ? `${r.rows / 1e6}M` : r.rows.toString();
    console.log(
      `${r.kernel.padEnd(17)} | ${r.build.padEnd(6)} | ${rowStr.padEnd(4)} | ${r.ops_per_sec.toLocaleString().padStart(14)} | ${r.ms_mean}ms`
    );
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
