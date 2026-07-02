/**
 * bench.mjs — Rust WASM spike benchmark (ADR-007)
 *
 * Identical protocol to the AS spike:
 *   - Kernels: add_f64, sum_f64_null, cmp_gt_f64_mask
 *   - Builds: simd, scalar
 *   - Row counts: 1_000_000, 10_000_000
 *   - Warmup: 3 runs; Measured: 10 (1M) or 5 (10M)
 *   - Reports: ops_per_sec, ms_mean
 *   - Also records gzipped .wasm sizes
 *
 * Run inside Docker:
 *   node wasm/spike/rust/bench.mjs
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DIST = join(__dirname, 'dist');

// ---------------------------------------------------------------------------
// Load a wasm module and return its exports + memory
// ---------------------------------------------------------------------------
async function loadWasm(name) {
  const wasmBytes = readFileSync(join(DIST, `${name}.wasm`));
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const exports = instance.exports;
  if (!exports.memory) {
    throw new Error(`${name}.wasm does not export 'memory'`);
  }
  return { exports, memory: exports.memory };
}

// ---------------------------------------------------------------------------
// Helpers to get typed array views over wasm memory
// ---------------------------------------------------------------------------
function f64View(memory, ptr, len) {
  return new Float64Array(memory.buffer, ptr, len);
}
function u8View(memory, ptr, len) {
  return new Uint8Array(memory.buffer, ptr, len);
}

// ---------------------------------------------------------------------------
// Fill an f64 array with deterministic data (values in range [0, 10))
// ---------------------------------------------------------------------------
function fillF64(view, seed = 42) {
  let x = seed >>> 0;
  for (let i = 0; i < view.length; i++) {
    // Simple xorshift32
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    // Map to [0, 10)
    view[i] = ((x >>> 0) / 0xFFFFFFFF) * 10.0;
  }
}

// ---------------------------------------------------------------------------
// Set all bits in a validity bitmap (all elements valid)
// ---------------------------------------------------------------------------
function fillAllValid(view) {
  view.fill(0xFF);
}

// ---------------------------------------------------------------------------
// Run one benchmark case
// ---------------------------------------------------------------------------
function runKernel(label, fn, warmup, measured) {
  // Warmup
  for (let i = 0; i < warmup; i++) fn();

  // Measured runs
  const times = [];
  for (let i = 0; i < measured; i++) {
    const t0 = performance.now();
    fn();
    times.push(performance.now() - t0);
  }

  const ms_mean = times.reduce((a, b) => a + b, 0) / times.length;
  return ms_mean;
}

// ---------------------------------------------------------------------------
// Main benchmark loop
// ---------------------------------------------------------------------------
async function main() {
  const builds = ['scalar', 'simd'];
  const rowCounts = [1_000_000, 10_000_000];
  const WARMUP = 3;

  // Load both wasm modules
  const modules = {};
  for (const build of builds) {
    console.log(`Loading ${build}.wasm...`);
    modules[build] = await loadWasm(build);
  }

  // Read gzipped sizes
  const sizes_gzip_bytes = {};
  for (const build of builds) {
    const bytes = readFileSync(join(DIST, `${build}.wasm`));
    const gz = gzipSync(bytes);
    sizes_gzip_bytes[build] = gz.length;
    console.log(`${build}.wasm: ${bytes.length} bytes raw, ${gz.length} bytes gzipped`);
  }

  // Read build metadata if available
  let build_time_s = {};
  try {
    const meta = JSON.parse(readFileSync(join(DIST, 'build_meta.json'), 'utf8'));
    build_time_s = {
      scalar: meta.scalar_build_ms / 1000,
      simd: meta.simd_build_ms / 1000,
    };
    console.log(`Build times: scalar=${build_time_s.scalar.toFixed(1)}s, simd=${build_time_s.simd.toFixed(1)}s`);
  } catch {
    console.log('build_meta.json not found, skipping build times');
  }

  const results = [];

  for (const build of builds) {
    const { exports, memory } = modules[build];
    const { alloc, add_f64, sum_f64_null, cmp_gt_f64_mask } = exports;

    for (const rows of rowCounts) {
      const measured = rows === 1_000_000 ? 10 : 5;
      console.log(`\n=== ${build} | ${(rows / 1e6).toFixed(0)}M rows (warmup=${WARMUP}, measured=${measured}) ===`);

      // --- Allocate buffers ---
      // Each allocation is persistent (bump allocator, no free)
      const bytesF64 = rows * 8;   // 8 bytes per f64
      const bytesMask = Math.ceil(rows / 8); // 1 bit per element

      const ptrA   = alloc(bytesF64);
      const ptrB   = alloc(bytesF64);
      const ptrOut = alloc(bytesF64);
      const ptrValidity = alloc(bytesMask);
      const ptrMask     = alloc(bytesMask);

      // Fill input data (re-create views after each potential memory.grow)
      let viewA = f64View(memory, ptrA, rows);
      let viewB = f64View(memory, ptrB, rows);
      fillF64(viewA, 42);
      fillF64(viewB, 137);

      const viewValidity = u8View(memory, ptrValidity, bytesMask);
      fillAllValid(viewValidity);

      // --- Kernel: add_f64 ---
      const ms_add = runKernel('add_f64', () => {
        add_f64(ptrA, ptrB, ptrOut, rows);
      }, WARMUP, measured);
      const ops_add = rows / (ms_add / 1000);
      console.log(`  add_f64:          ${ms_add.toFixed(2)} ms/run  ${(ops_add / 1e6).toFixed(1)} Mops/s`);
      results.push({ kernel: 'add_f64', build, rows, ops_per_sec: Math.round(ops_add), ms_mean: +ms_add.toFixed(3) });

      // --- Kernel: sum_f64_null ---
      let lastSum = 0;
      const ms_sum = runKernel('sum_f64_null', () => {
        lastSum = sum_f64_null(ptrA, ptrValidity, rows);
      }, WARMUP, measured);
      const ops_sum = rows / (ms_sum / 1000);
      console.log(`  sum_f64_null:     ${ms_sum.toFixed(2)} ms/run  ${(ops_sum / 1e6).toFixed(1)} Mops/s  (sum=${lastSum.toFixed(2)})`);
      results.push({ kernel: 'sum_f64_null', build, rows, ops_per_sec: Math.round(ops_sum), ms_mean: +ms_sum.toFixed(3) });

      // --- Kernel: cmp_gt_f64_mask ---
      // Use 5.0 as scalar (roughly half the data will match since data is [0,10))
      const scalar = 5.0;
      const ms_cmp = runKernel('cmp_gt_f64_mask', () => {
        cmp_gt_f64_mask(ptrA, scalar, ptrMask, rows);
      }, WARMUP, measured);
      const ops_cmp = rows / (ms_cmp / 1000);
      console.log(`  cmp_gt_f64_mask:  ${ms_cmp.toFixed(2)} ms/run  ${(ops_cmp / 1e6).toFixed(1)} Mops/s`);
      results.push({ kernel: 'cmp_gt_f64_mask', build, rows, ops_per_sec: Math.round(ops_cmp), ms_mean: +ms_cmp.toFixed(3) });
    }
  }

  // ---------------------------------------------------------------------------
  // Write results.json
  // ---------------------------------------------------------------------------
  const output = {
    results,
    sizes_gzip_bytes,
    build_time_s,
    meta: {
      node_version: process.version,
      date: new Date().toISOString(),
      warmup_runs: WARMUP,
      measured_runs: { '1000000': 10, '10000000': 5 },
    },
  };

  const resultsPath = join(__dirname, 'results.json');
  writeFileSync(resultsPath, JSON.stringify(output, null, 2));
  console.log(`\n=== Results written to ${resultsPath} ===`);

  // Print summary table
  console.log('\n--- Summary (ops/sec in millions) ---');
  console.log('kernel              | build  | rows  | Mops/s | ms_mean');
  console.log('--------------------|--------|-------|--------|--------');
  for (const r of results) {
    const k = r.kernel.padEnd(19);
    const b = r.build.padEnd(6);
    const rows = (r.rows / 1e6).toFixed(0).padStart(5);
    const ops = (r.ops_per_sec / 1e6).toFixed(1).padStart(6);
    const ms = r.ms_mean.toFixed(2).padStart(7);
    console.log(`${k} | ${b} | ${rows}M | ${ops} | ${ms}`);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
