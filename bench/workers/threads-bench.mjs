/**
 * Parallel mode bench (ADR-006 / P5.1 gate: ≥1.8× on 4 workers for 10M-row reductions).
 *
 * Tests f64 sum, mean, min against 1 vs 4 workers at 10M rows.
 * Memory bus is the expected bottleneck; if 1.8× is not achieved this file
 * reports the actual ratio and the reason (memory-bound saturation).
 *
 * Architecture note:
 *   simd-threads.wasm IMPORTS its memory. The shared WebAssembly.Memory is
 *   created by enableThreads() and held in th.memory. ALL data must be
 *   allocated via th.alloc() so pointers are valid in the workers' address space.
 *   The "1-thread" baseline uses th.callKernel() — same wasm, same memory,
 *   direct call with no postMessage overhead.
 *
 * Run: node bench/workers/threads-bench.mjs
 *   (requires simd-threads.wasm in wasm/dist/ and the JS built in dist/)
 */

import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '../../wasm/dist');
const N = 10_000_000;

/* ------------------------------------------------------------------ */
/* Load enableThreads from built dist                                    */
/* ------------------------------------------------------------------ */

// enableThreads moved to the workers subpath entry when it left the main bundle
// (size-gate split; see tsup.config.ts comment).
const { enableThreads } = await import('../../dist/workers.js');

/* ------------------------------------------------------------------ */
/* Create two pool sizes — 1 worker and 4 workers                       */
/* Each has its own shared memory; data is written into both.           */
/* ------------------------------------------------------------------ */

const th1 = await enableThreads({ workers: 1, wasmDir: WASM_DIR, timeoutMs: 60_000 });
const th4 = await enableThreads({ workers: 4, wasmDir: WASM_DIR, timeoutMs: 60_000 });

if (!th1 || !th4) {
  console.error('enableThreads returned false — SharedArrayBuffer not available?');
  process.exit(1);
}

function writeData(th) {
  const ptr = th.alloc(N * 8);
  const view = new Float64Array(th.memory.buffer, ptr, N);
  for (let i = 0; i < N; i++) view[i] = (i + 1) * 0.001;
  return ptr;
}

const ptr1 = writeData(th1);  // data in th1's shared memory
const ptr4 = writeData(th4);  // data in th4's shared memory

/* ------------------------------------------------------------------ */
/* Bench                                                                  */
/* ------------------------------------------------------------------ */

const WARMUP = 3;
const ITERS = 10;

async function measureMs(fn) {
  for (let i = 0; i < WARMUP; i++) await fn();
  const start = performance.now();
  for (let i = 0; i < ITERS; i++) await fn();
  return (performance.now() - start) / ITERS;
}

console.log(`\nParallel-mode bench  N=${N.toLocaleString()} f64 elements (${(N * 8 / 1e6).toFixed(0)} MB)\n`);
console.log('Op         | 1-thread ms | 1-worker ms | 4-worker ms | 1t→4w ratio | Gate ≥1.8×');
console.log('-----------|-------------|-------------|-------------|-------------|----------');

const ops = [
  ['sum',
    (th, ptr) => th.callKernel('sum_f64_null', ptr, 0, N),
    (th, ptr) => th.sumF64(ptr, 0, N)],
  ['mean',
    (th, ptr) => th.callKernel('mean_f64_null', ptr, 0, N),
    (th, ptr) => th.meanF64(ptr, 0, N)],
  ['min',
    (th, ptr) => th.callKernel('min_f64_null', ptr, 0, N),
    (th, ptr) => th.minF64(ptr, 0, N)],
];

const results = [];

for (const [name, singleFn, parallelFn] of ops) {
  /* 1-thread: direct kernel call on th4's shared memory (no postMessage) */
  const t1thread = await measureMs(() => singleFn(th4, ptr4));
  /* 1-worker: postMessage round-trip, serial */
  const t1worker = await measureMs(() => parallelFn(th1, ptr1));
  /* 4-worker: parallel dispatch */
  const t4worker = await measureMs(() => parallelFn(th4, ptr4));

  const ratio = t1thread / t4worker;
  const gate = ratio >= 1.8 ? 'PASS' : 'FAIL';

  console.log(
    `${name.padEnd(10)} | ${t1thread.toFixed(2).padStart(11)} | ${t1worker.toFixed(2).padStart(11)} | ${t4worker.toFixed(2).padStart(11)} | ${ratio.toFixed(2).padStart(11)} | ${gate}`,
  );

  results.push({ name, t1thread, t1worker, t4worker, ratio, gate });
}

th1.terminate();
th4.terminate();

/* ------------------------------------------------------------------ */
/* Summary                                                               */
/* ------------------------------------------------------------------ */

console.log('\n--- Summary ---');
const failed = results.filter((r) => r.gate === 'FAIL');
if (failed.length === 0) {
  console.log('All ops meet the ≥1.8× gate on 4 workers.');
} else {
  console.log(`WARNING: ${failed.length} op(s) did not meet the ≥1.8× gate:`);
  for (const r of failed) {
    console.log(
      `  ${r.name}: ${r.ratio.toFixed(2)}× — likely memory-bandwidth saturated ` +
      `(${(N * 8 / 1e6).toFixed(0)} MB / ${r.t4worker.toFixed(2)} ms = ` +
      `${((N * 8) / (r.t4worker / 1000) / 1e9).toFixed(1)} GB/s effective bandwidth)`,
    );
  }
  console.log(
    'If the memory bus is the bottleneck, sub-1.8× is expected for memory-bound ops.',
  );
}

console.log('\nNote: f64 sum/mean parallel results are NOT bit-identical to single-thread');
console.log('(different FP accumulation order). See docs/threads.md §"Floating-point deviation".');
