/**
 * Benchmark: hash kernel family (Agent D) vs TypedArray-JS baseline.
 *
 * Headline comparison (spec §5 gate: ≥ 1.5× for kernels with a JS analog):
 *   - group_build + hash_i32 vs JS Map-based groupby  @1M rows, 10K uniques
 *   - join_hash_inner vs JS Map-based inner join       @1M × 100K rows
 *
 * Additional absolute-number benchmarks:
 *   - hash_i32 alone @1M rows
 *   - hash_combine @1M rows
 *   - join_hash_left @1M × 100K rows
 *
 * Usage:
 *   node bench/kernels/hash.mjs
 *
 * Outputs a results table to stdout. The gate ratio is printed at the end.
 *
 * Hash function used by WASM kernels: splitmix64 (documented in wasm/rust/src/hash.rs).
 */

import { Bench } from 'tinybench';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { mulberry32 } from '../datasets.mjs';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '../../wasm/dist');

// ---------------------------------------------------------------------------
// WASM loader (load both scalar and simd builds)
// ---------------------------------------------------------------------------

async function loadWasm(simd) {
  const fileName = simd ? 'simd.wasm' : 'scalar.wasm';
  const filePath = join(WASM_DIR, fileName);
  const bytes = await readFile(filePath);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------

function allocI32(ex, vals) {
  const ptr = ex.alloc(vals.length * 4);
  new Int32Array(ex.memory.buffer, ptr, vals.length).set(vals);
  return ptr;
}

function allocI64FromBigInt(ex, vals) {
  const ptr = ex.alloc(vals.length * 8);
  new BigInt64Array(ex.memory.buffer, ptr, vals.length).set(vals);
  return ptr;
}

function allocZeroed(ex, bytes) {
  const ptr = ex.alloc(bytes);
  new Uint8Array(ex.memory.buffer, ptr, bytes).fill(0);
  return ptr;
}

function nextPow2(n) {
  if (n <= 1) return 1;
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

// ---------------------------------------------------------------------------
// Dataset generation
// ---------------------------------------------------------------------------

const N = 1_000_000;    // probe / groupby size
const N_JOIN_RIGHT = 100_000; // right (build) side for joins
const G_UNIQUE = 10_000; // unique groups for groupby

console.log(`Generating datasets: N=${N.toLocaleString()}, groups=${G_UNIQUE.toLocaleString()}, join-right=${N_JOIN_RIGHT.toLocaleString()}`);

const rng = mulberry32(0xdeadbeef);

// i32 column with 10K unique values (for groupby)
const groupData = new Int32Array(N);
for (let i = 0; i < N; i++) {
  groupData[i] = (rng() * G_UNIQUE) | 0;
}

// i32 column for join left (1M rows, values 0..99999)
const joinLeft = new Int32Array(N);
const rngL = mulberry32(0xcafe);
for (let i = 0; i < N; i++) {
  joinLeft[i] = (rngL() * N_JOIN_RIGHT) | 0;
}

// i32 column for join right (100K rows, values 0..99999 — dense keys)
const joinRight = new Int32Array(N_JOIN_RIGHT);
for (let i = 0; i < N_JOIN_RIGHT; i++) {
  joinRight[i] = i; // all distinct keys
}

// ---------------------------------------------------------------------------
// JS TypedArray baseline implementations
// ---------------------------------------------------------------------------

/** JS Map-based groupby: assign group IDs to i32 data with 10K unique values. */
function jsGroupby(data) {
  const n = data.length;
  const map = new Map();
  let groupCount = 0;
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const v = data[i];
    let g = map.get(v);
    if (g === undefined) {
      g = groupCount++;
      map.set(v, g);
    }
    out[i] = g;
  }
  return { groupCount, out };
}

/** JS Map-based inner join: build on right, probe left. */
function jsJoinInner(leftData, rightData) {
  // Build phase: right side into a Map (key → list of right indices)
  const buildMap = new Map();
  for (let r = 0; r < rightData.length; r++) {
    const v = rightData[r];
    if (!buildMap.has(v)) buildMap.set(v, []);
    buildMap.get(v).push(r);
  }
  // Probe phase
  const outL = [];
  const outR = [];
  for (let l = 0; l < leftData.length; l++) {
    const matches = buildMap.get(leftData[l]);
    if (matches) {
      for (const r of matches) {
        outL.push(l);
        outR.push(r);
      }
    }
  }
  return { count: outL.length };
}

// ---------------------------------------------------------------------------
// WASM kernel wrappers (using the retry protocol)
// ---------------------------------------------------------------------------

/**
 * WASM hash_i32 + group_build in one call.
 * Returns groupCount. The out buffer is reused across calls.
 */
function wasmGroupby(ex, dataPtr, len, outHashPtr, outGroupIds) {
  // Step 1: hash the i32 column
  ex.hash_i32(dataPtr, 0 /* all-valid */, outHashPtr, len);

  // Step 2: group_build with retry
  let htCap = nextPow2(2 * Math.max(len, 1));
  const MIN_HT_CAP = 4;
  if (htCap < MIN_HT_CAP) htCap = MIN_HT_CAP;

  let htPtr = 0;
  try {
    htPtr = allocZeroed(ex, htCap * 16);
    for (;;) {
      const result = ex.group_build(outHashPtr, len, htPtr, htCap, outGroupIds);
      if (result !== -1) {
        return result;
      }
      ex.free(htPtr); htPtr = 0;
      htCap *= 2;
      htPtr = allocZeroed(ex, htCap * 16);
    }
  } finally {
    if (htPtr !== 0) ex.free(htPtr);
  }
}

/**
 * WASM join_hash_inner with retry.
 */
function wasmJoinInner(ex, lhPtr, lLen, rhPtr, rLen, outLPtr, outRPtr) {
  let htCap = nextPow2(2 * Math.max(rLen, 1));
  if (htCap < 4) htCap = 4;
  let outCap = Math.max(1, lLen + rLen);

  let htPtr = 0;
  let outLLocal = outLPtr;
  let outRLocal = outRPtr;
  let ownOut = false;
  try {
    htPtr = allocZeroed(ex, htCap * 16);
    for (;;) {
      const n = ex.join_hash_inner(
        lhPtr, 0, lLen, rhPtr, 0, rLen,
        htPtr, htCap, outLLocal, outRLocal, outCap,
      );
      if (n === -1) {
        ex.free(htPtr); htPtr = 0;
        htCap *= 2;
        htPtr = allocZeroed(ex, htCap * 16);
        continue;
      }
      if (n > outCap) {
        if (ownOut) { ex.free(outLLocal); ex.free(outRLocal); }
        outCap = n;
        outLLocal = ex.alloc(outCap * 4);
        outRLocal = ex.alloc(outCap * 4);
        ownOut = true;
        ex.free(htPtr); htPtr = 0;
        htPtr = allocZeroed(ex, htCap * 16);
        continue;
      }
      return n;
    }
  } finally {
    if (htPtr !== 0) ex.free(htPtr);
    if (ownOut) { ex.free(outLLocal); ex.free(outRLocal); }
  }
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

async function main() {
  console.log('\n=== Loading WASM builds ===');
  const [scalarEx, simdEx] = await Promise.all([
    loadWasm(false),
    loadWasm(true),
  ]);

  // Pre-allocate persistent WASM buffers (avoid per-iteration alloc overhead)
  const groupDataPtr = allocI32(scalarEx, groupData);
  const outHashPtrS = scalarEx.alloc(N * 8);
  const outGroupIdsPtrS = scalarEx.alloc(N * 4);

  const groupDataPtrM = allocI32(simdEx, groupData);
  const outHashPtrM = simdEx.alloc(N * 8);
  const outGroupIdsPtrM = simdEx.alloc(N * 4);

  const joinLPtrS = allocI32(scalarEx, joinLeft);
  const joinRPtrS = allocI32(scalarEx, joinRight);
  const joinLHashPtrS = scalarEx.alloc(N * 8);
  const joinRHashPtrS = scalarEx.alloc(N_JOIN_RIGHT * 8);
  // Pre-compute join hashes (hash is fast; join build/probe is the main cost)
  scalarEx.hash_i32(joinLPtrS, 0, joinLHashPtrS, N);
  scalarEx.hash_i32(joinRPtrS, 0, joinRHashPtrS, N_JOIN_RIGHT);
  const joinOutLPtrS = scalarEx.alloc(N * 4);
  const joinOutRPtrS = scalarEx.alloc(N * 4);

  const joinLPtrM = allocI32(simdEx, joinLeft);
  const joinRPtrM = allocI32(simdEx, joinRight);
  const joinLHashPtrM = simdEx.alloc(N * 8);
  const joinRHashPtrM = simdEx.alloc(N_JOIN_RIGHT * 8);
  simdEx.hash_i32(joinLPtrM, 0, joinLHashPtrM, N);
  simdEx.hash_i32(joinRPtrM, 0, joinRHashPtrM, N_JOIN_RIGHT);
  const joinOutLPtrM = simdEx.alloc(N * 4);
  const joinOutRPtrM = simdEx.alloc(N * 4);

  console.log('WASM allocations ready.\n');

  // --- Bench config ---
  const BENCH_MS = 3000;
  const WARMUP = 3;

  const bench = new Bench({ time: BENCH_MS, warmupIterations: WARMUP });

  // --- Groupby benchmarks ---

  bench.add('JS Map groupby @1M (10K uniques)', () => {
    jsGroupby(groupData);
  });

  bench.add('WASM scalar hash_i32+group_build @1M (10K uniques)', () => {
    wasmGroupby(scalarEx, groupDataPtr, N, outHashPtrS, outGroupIdsPtrS);
  });

  bench.add('WASM simd hash_i32+group_build @1M (10K uniques)', () => {
    wasmGroupby(simdEx, groupDataPtrM, N, outHashPtrM, outGroupIdsPtrM);
  });

  // --- hash_i32 alone (absolute number) ---

  bench.add('WASM scalar hash_i32 only @1M', () => {
    scalarEx.hash_i32(groupDataPtr, 0, outHashPtrS, N);
  });

  bench.add('WASM simd hash_i32 only @1M', () => {
    simdEx.hash_i32(groupDataPtrM, 0, outHashPtrM, N);
  });

  // --- hash_combine alone (absolute number) ---

  bench.add('WASM scalar hash_combine @1M', () => {
    // Use outHashPtr as both acc and add for a consistent workload.
    scalarEx.hash_combine(outHashPtrS, groupDataPtr, N);
  });

  // --- Join benchmarks ---

  bench.add('JS Map join inner @1Mx100K', () => {
    jsJoinInner(joinLeft, joinRight);
  });

  bench.add('WASM scalar join_hash_inner @1Mx100K', () => {
    wasmJoinInner(scalarEx, joinLHashPtrS, N, joinRHashPtrS, N_JOIN_RIGHT,
      joinOutLPtrS, joinOutRPtrS);
  });

  bench.add('WASM simd join_hash_inner @1Mx100K', () => {
    wasmJoinInner(simdEx, joinLHashPtrM, N, joinRHashPtrM, N_JOIN_RIGHT,
      joinOutLPtrM, joinOutRPtrM);
  });

  console.log('=== Running benchmarks (this will take ~30s) ===\n');
  await bench.run();

  // --- Results table ---
  console.log('\n=== Results ===');
  console.log(
    'Name'.padEnd(52) +
    'Mean (ms)'.padStart(12) +
    'Ops/s'.padStart(12),
  );
  console.log('-'.repeat(76));

  const results = {};
  for (const task of bench.tasks) {
    const ms = task.result.mean;
    const ops = 1000 / ms;
    results[task.name] = { ms, ops };
    console.log(
      task.name.padEnd(52) +
      ms.toFixed(3).padStart(12) +
      ops.toFixed(1).padStart(12),
    );
  }

  // --- Gate ratios ---
  console.log('\n=== Gate Ratios (§5 target: ≥ 1.5× over JS analog) ===');

  const jsGroupbyMs = results['JS Map groupby @1M (10K uniques)']?.ms ?? NaN;
  const wasmScalarGroupbyMs = results['WASM scalar hash_i32+group_build @1M (10K uniques)']?.ms ?? NaN;
  const wasmSimdGroupbyMs = results['WASM simd hash_i32+group_build @1M (10K uniques)']?.ms ?? NaN;
  const jsJoinMs = results['JS Map join inner @1Mx100K']?.ms ?? NaN;
  const wasmScalarJoinMs = results['WASM scalar join_hash_inner @1Mx100K']?.ms ?? NaN;
  const wasmSimdJoinMs = results['WASM simd join_hash_inner @1Mx100K']?.ms ?? NaN;

  const scalarGroupbyRatio = jsGroupbyMs / wasmScalarGroupbyMs;
  const simdGroupbyRatio = jsGroupbyMs / wasmSimdGroupbyMs;
  const scalarJoinRatio = jsJoinMs / wasmScalarJoinMs;
  const simdJoinRatio = jsJoinMs / wasmSimdJoinMs;

  const gate = (label, ratio) => {
    const pass = ratio >= 1.5;
    const mark = pass ? '✓ PASS' : '✗ FAIL';
    console.log(`  ${mark}  ${label}: ${ratio.toFixed(2)}× (${pass ? 'above' : 'below'} 1.5× gate)`);
  };

  gate('WASM scalar groupby vs JS Map', scalarGroupbyRatio);
  gate('WASM simd   groupby vs JS Map', simdGroupbyRatio);
  gate('WASM scalar join_inner vs JS Map-join', scalarJoinRatio);
  gate('WASM simd   join_inner vs JS Map-join', simdJoinRatio);

  // Absolute numbers for non-gated kernels
  console.log('\n=== Absolute numbers (no JS analog) ===');
  const hashI32ScalarMs = results['WASM scalar hash_i32 only @1M']?.ms ?? NaN;
  const hashI32SimdMs = results['WASM simd hash_i32 only @1M']?.ms ?? NaN;
  const hashCombineScalarMs = results['WASM scalar hash_combine @1M']?.ms ?? NaN;
  console.log(`  hash_i32 scalar @1M: ${hashI32ScalarMs.toFixed(3)} ms`);
  console.log(`  hash_i32 simd   @1M: ${hashI32SimdMs.toFixed(3)} ms`);
  console.log(`  hash_combine scalar @1M: ${hashCombineScalarMs.toFixed(3)} ms`);

  // Overall gate summary
  const allPass = [scalarGroupbyRatio, simdGroupbyRatio, scalarJoinRatio, simdJoinRatio]
    .every((r) => r >= 1.5);
  console.log(`\n=== Gate summary: ${allPass ? 'ALL PASS ✓' : 'SOME FAILED ✗'} ===\n`);

  // Return exit code for CI
  process.exitCode = allPass ? 0 : 1;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
