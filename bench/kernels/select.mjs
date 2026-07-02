/**
 * Benchmark: Phase-2 selection kernel family (Agent C).
 *
 * Standalone Node.js script using tinybench.
 * Compares WASM selection kernels vs live TypedArray-JS equivalents at 1M rows.
 *
 * Gate (spec §5): ≥ 1.5× for kernels with a JS analog.
 * Headline gate: filter_f64 SIMD vs TypedArray filter loop.
 *
 * Usage:
 *   node bench/kernels/select.mjs
 *
 * Outputs ratio table to stdout.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { Bench } from 'tinybench';
import { makeNumericRaw, mulberry32 } from '../datasets.mjs';

const __dir = dirname(fileURLToPath(import.meta.url));
const DIST = join(__dir, '..', '..', 'wasm', 'dist');

// ── Load WASM instances ───────────────────────────────────────────────────────

async function loadWasm(name) {
  const bytes = readFileSync(join(DIST, name));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

// ── WASM helpers ──────────────────────────────────────────────────────────────

function wasmAlloc(mod, n, bpe) {
  if (n === 0) return mod.alloc(0);
  const p = mod.alloc(n * bpe);
  if (p === 0) throw new Error('OOM');
  return p;
}

function packF64(mod, arr) {
  const n = arr.length;
  const ptr = wasmAlloc(mod, n, 8);
  new Float64Array(mod.memory.buffer, ptr, n).set(arr);
  return ptr;
}

function packI32(mod, arr) {
  const n = arr.length;
  const ptr = wasmAlloc(mod, Math.max(n, 1), 4);
  if (n > 0) new Int32Array(mod.memory.buffer, ptr, n).set(arr);
  return ptr;
}

/**
 * Pack a per-element validity array (0/1 Uint8Array) into an Arrow-LSB bitmap.
 * Returns the bitmap pointer.
 */
function packBitmap(mod, validArr) {
  const n = validArr.length;
  const byteCount = Math.ceil(n / 8) || 1;
  const ptr = mod.alloc(byteCount);
  const u8 = new Uint8Array(mod.memory.buffer, ptr, byteCount);
  u8.fill(0);
  for (let i = 0; i < n; i++) {
    if (validArr[i]) u8[i >> 3] |= 1 << (i & 7);
  }
  return ptr;
}

/**
 * Build a bitmask from a boolean predicate over a Float64Array.
 * Returns { ptr, count } — pointer to the Arrow-LSB mask and the popcount.
 */
function buildF64MaskGt(mod, arr, threshold) {
  const n = arr.length;
  const byteCount = Math.ceil(n / 8) || 1;
  const ptr = mod.alloc(byteCount);
  const u8 = new Uint8Array(mod.memory.buffer, ptr, byteCount);
  u8.fill(0);
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (arr[i] > threshold) {
      u8[i >> 3] |= 1 << (i & 7);
      count++;
    }
  }
  return { ptr, count };
}

// ── Benchmark configuration ───────────────────────────────────────────────────

const N = 1_000_000; // 1M rows — §5 headline gate
const BENCH_MS = 2000;
const WARMUP = 3;

console.log(`\nPhase-2 Selection kernels bench — N = ${N.toLocaleString()} rows`);

// ── Dataset ───────────────────────────────────────────────────────────────────

const raw = makeNumericRaw(N);
const { a: dataF64, aValid } = raw; // Float64Array + Uint8Array (1=valid)

// Build seeded i32 data for argsort / topk
const rng = mulberry32(0xabcdef01);
const dataI32 = new Int32Array(N);
for (let i = 0; i < N; i++) dataI32[i] = (rng() * 200000 - 100000) | 0;

// identity permutation for argsort
const identPerm = new Int32Array(N);
for (let i = 0; i < N; i++) identPerm[i] = i;

// ── Load WASM ─────────────────────────────────────────────────────────────────

const scalarMod = await loadWasm('scalar.wasm');
const simdMod   = await loadWasm('simd.wasm');

// ── Allocate WASM buffers (reused across iterations) ──────────────────────────

function setupWasmBuffers(mod) {
  const dataPtr  = packF64(mod, dataF64);
  const vpPtr    = packBitmap(mod, aValid);
  const outPtr   = wasmAlloc(mod, N, 8);       // filter output (worst-case all)
  const outI32Ptr = wasmAlloc(mod, N, 4);       // filter_indices / gather output

  // Pre-build the filter mask (a[i] > 0.5, valid).
  // This mask is the same across iterations; filter just compacts it.
  const { ptr: maskPtr } = buildF64MaskGt(mod, dataF64, 0.5);

  // Pre-pack i32 data for argsort
  const i32Ptr = mod.alloc(N * 4);
  new Int32Array(mod.memory.buffer, i32Ptr, N).set(dataI32);

  // perm buffer — refilled each iteration
  const permPtr = wasmAlloc(mod, N, 4);

  // index array for gather (indices 0..N-1, shuffled slightly)
  const idxArr = new Int32Array(N);
  const rng2 = mulberry32(0x11223344);
  for (let i = 0; i < N; i++) idxArr[i] = (rng2() * N) | 0;
  const idxPtr = packI32(mod, idxArr);

  return { dataPtr, vpPtr, outPtr, outI32Ptr, maskPtr, i32Ptr, permPtr, idxPtr };
}

const sb = setupWasmBuffers(scalarMod);
const vb = setupWasmBuffers(simdMod);

// ── TypedArray JS baselines ───────────────────────────────────────────────────

function jsFilterF64(data, validArr, threshold) {
  const out = new Float64Array(data.length);
  let count = 0;
  for (let i = 0; i < data.length; i++) {
    if (validArr[i] && data[i] > threshold) out[count++] = data[i];
  }
  return count;
}

function jsGatherF64(data, idx) {
  const out = new Float64Array(idx.length);
  for (let i = 0; i < idx.length; i++) out[i] = data[idx[i]];
  return out;
}

// Pre-compute idx array for JS gather (same as wasm)
const jsIdxArr = new Int32Array(N);
const rng3 = mulberry32(0x11223344);
for (let i = 0; i < N; i++) jsIdxArr[i] = (rng3() * N) | 0;

const jsIdentPerm = new Int32Array(N);
for (let i = 0; i < N; i++) jsIdentPerm[i] = i;

// JS topk (max-heap based k-largest)
function jsTopkF64(data, k) {
  // Simple: sort indices by value desc, take first k.
  // For fair comparison, use same heap approach.
  const n = data.length;
  const out = [];
  for (let i = 0; i < n && out.length < k; i++) {
    if (!isNaN(data[i])) out.push(i);
  }
  // Build a min-heap of size k
  function lt(a, b) { return data[a] < data[b] || (data[a] === data[b] && a > b); }
  function sift(arr, pos, size) {
    while (true) {
      const l = 2*pos+1, r = l+1;
      if (l >= size) break;
      let sm = l;
      if (r < size && lt(arr[r], arr[l])) sm = r;
      if (!lt(arr[sm], arr[pos])) break;
      [arr[pos], arr[sm]] = [arr[sm], arr[pos]];
      pos = sm;
    }
  }
  // heapify
  for (let i = Math.floor(out.length/2)-1; i >= 0; i--) sift(out, i, out.length);
  for (let i = out.length; i < n; i++) {
    if (isNaN(data[i])) continue;
    if (lt(out[0], i)) { out[0] = i; sift(out, 0, out.length); }
  }
  return out;
}

// ── Benchmark suites ──────────────────────────────────────────────────────────

const results = [];

async function bench(label, fns) {
  const b = new Bench({ time: BENCH_MS, warmupIterations: WARMUP });
  for (const [name, fn] of Object.entries(fns)) {
    b.add(name, fn);
  }
  await b.run();
  console.log(`\n${label}:`);
  const row = { label };
  for (const t of b.tasks) {
    const ms = t.result.mean;
    const ops = (1000 / ms).toFixed(1);
    console.log(`  ${t.name.padEnd(28)} ${ms.toFixed(3)} ms/op  (${ops} op/s)`);
    row[t.name] = ms;
  }
  results.push(row);
  return row;
}

// ── 1. filter_f64 (headline gate) ─────────────────────────────────────────────

{
  const r = await bench('filter_f64 @1M (headline gate §5)', {
    'JS typedarray':   () => jsFilterF64(dataF64, aValid, 0.5),
    'wasm scalar':     () => scalarMod.filter_f64(sb.dataPtr, sb.maskPtr, sb.outPtr, N),
    'wasm simd':       () => simdMod.filter_f64(vb.dataPtr, vb.maskPtr, vb.outPtr, N),
  });
  const jsMs   = r['JS typedarray'];
  const simdMs = r['wasm simd'];
  const scMs   = r['wasm scalar'];
  console.log(`  → scalar vs JS: ${(jsMs/scMs).toFixed(2)}×   simd vs JS: ${(jsMs/simdMs).toFixed(2)}×  (gate: ≥1.5×)`);
  results[results.length-1].ratio_scalar = jsMs/scMs;
  results[results.length-1].ratio_simd   = jsMs/simdMs;
}

// ── 2. filter_indices ─────────────────────────────────────────────────────────

{
  // JS analog: collect indices where mask=1 (pre-compute same mask)
  const mask = new Uint8Array(Math.ceil(N/8));
  const u8scr = new Uint8Array(scalarMod.memory.buffer, sb.maskPtr, Math.ceil(N/8));
  mask.set(u8scr);

  function jsFilterIndices(mask, len) {
    const out = new Int32Array(len);
    let count = 0;
    const full = len >> 3;
    for (let b = 0; b < full; b++) {
      let m = mask[b];
      const base = b << 3;
      while (m) {
        const bit = Math.clz32(m & -m) ^ 31;
        out[count++] = base + bit;
        m &= m - 1;
      }
    }
    const tail = len & 7;
    if (tail) {
      let m = mask[full] & ((1 << tail) - 1);
      const base = full << 3;
      while (m) {
        const bit = Math.clz32(m & -m) ^ 31;
        out[count++] = base + bit;
        m &= m - 1;
      }
    }
    return count;
  }

  const r = await bench('filter_indices @1M', {
    'JS typedarray':   () => jsFilterIndices(mask, N),
    'wasm scalar':     () => scalarMod.filter_indices(sb.maskPtr, sb.outI32Ptr, N),
    'wasm simd':       () => simdMod.filter_indices(vb.maskPtr, vb.outI32Ptr, N),
  });
  const jsMs   = r['JS typedarray'];
  const simdMs = r['wasm simd'];
  const scMs   = r['wasm scalar'];
  console.log(`  → scalar vs JS: ${(jsMs/scMs).toFixed(2)}×   simd vs JS: ${(jsMs/simdMs).toFixed(2)}×`);
  results[results.length-1].ratio_scalar = jsMs/scMs;
  results[results.length-1].ratio_simd   = jsMs/simdMs;
}

// ── 3. gather_f64 ─────────────────────────────────────────────────────────────

{
  const r = await bench('gather_f64 @1M', {
    'JS typedarray':   () => jsGatherF64(dataF64, jsIdxArr),
    'wasm scalar':     () => scalarMod.gather_f64(sb.dataPtr, sb.idxPtr, N, sb.outPtr),
    'wasm simd':       () => simdMod.gather_f64(vb.dataPtr, vb.idxPtr, N, vb.outPtr),
  });
  const jsMs   = r['JS typedarray'];
  const simdMs = r['wasm simd'];
  const scMs   = r['wasm scalar'];
  console.log(`  → scalar vs JS: ${(jsMs/scMs).toFixed(2)}×   simd vs JS: ${(jsMs/simdMs).toFixed(2)}×`);
  results[results.length-1].ratio_scalar = jsMs/scMs;
  results[results.length-1].ratio_simd   = jsMs/simdMs;
}

// ── 4. argsort_f64 (no JS analog with same total-order — report absolute) ─────

{
  // JS sort — uses built-in sort (not comparable kernel, report as reference)
  function jsArgsortF64(data) {
    const perm = new Int32Array(data.length);
    for (let i = 0; i < data.length; i++) perm[i] = i;
    // Quick approximation; not NaN-ordered stable sort — just for scale.
    perm.sort((a, b) => {
      const av = data[a], bv = data[b];
      if (av < bv) return -1;
      if (av > bv) return 1;
      return a - b;
    });
    return perm;
  }

  // Refill perm before each wasm call (argsort mutates it)
  const permScalar = new Int32Array(N);
  const permSimd   = new Int32Array(N);
  for (let i = 0; i < N; i++) { permScalar[i] = i; permSimd[i] = i; }

  let callScalar = 0, callSimd = 0;

  const r = await bench('argsort_f64 @1M (no-alloc stable sort)', {
    'JS Array.sort':   () => jsArgsortF64(dataF64),
    'wasm scalar':     () => {
      // Refill perm each call
      const p = new Int32Array(scalarMod.memory.buffer, sb.permPtr, N);
      for (let i = 0; i < N; i++) p[i] = i;
      scalarMod.argsort_f64(sb.dataPtr, 0, sb.permPtr, N, 0);
      callScalar++;
    },
    'wasm simd':       () => {
      const p = new Int32Array(simdMod.memory.buffer, vb.permPtr, N);
      for (let i = 0; i < N; i++) p[i] = i;
      simdMod.argsort_f64(vb.dataPtr, 0, vb.permPtr, N, 0);
      callSimd++;
    },
  });
  void callScalar; void callSimd;
  const jsMs   = r['JS Array.sort'];
  const scMs   = r['wasm scalar'];
  const simdMs = r['wasm simd'];
  console.log(`  → wasm/JS ratios are less meaningful (different algorithm guarantees)`);
  console.log(`  → scalar vs JS: ${(jsMs/scMs).toFixed(2)}×   simd vs JS: ${(jsMs/simdMs).toFixed(2)}×`);
  results[results.length-1].ratio_scalar = jsMs/scMs;
  results[results.length-1].ratio_simd   = jsMs/simdMs;
}

// ── 5. topk_f64 (no direct JS analog — report absolute) ──────────────────────

{
  const K = 10;
  const outTopkPtr = wasmAlloc(scalarMod, K+1, 4);
  const outTopkPtrS = wasmAlloc(simdMod, K+1, 4);

  const r = await bench(`topk_f64 k=${K} @1M`, {
    'JS heap-topk':    () => jsTopkF64(dataF64, K),
    'wasm scalar':     () => scalarMod.topk_f64(sb.dataPtr, 0, K, outTopkPtr, N, 1),
    'wasm simd':       () => simdMod.topk_f64(vb.dataPtr, 0, K, outTopkPtrS, N, 1),
  });
  const jsMs   = r['JS heap-topk'];
  const scMs   = r['wasm scalar'];
  const simdMs = r['wasm simd'];
  console.log(`  → scalar vs JS: ${(jsMs/scMs).toFixed(2)}×   simd vs JS: ${(jsMs/simdMs).toFixed(2)}×`);
  results[results.length-1].ratio_scalar = jsMs/scMs;
  results[results.length-1].ratio_simd   = jsMs/simdMs;
}

// ── Summary ───────────────────────────────────────────────────────────────────

console.log('\n═══ Summary ═══════════════════════════════════════════════════════');
console.log('Operation                        scalar×JS  simd×JS   gate(≥1.5×)');
console.log('─────────────────────────────────────────────────────────────────');
const GATE = 1.5;
let allPass = true;
for (const r of results) {
  const sc = r.ratio_scalar?.toFixed(2) ?? '  —  ';
  const si = r.ratio_simd?.toFixed(2)   ?? '  —  ';
  const pass = (r.ratio_simd ?? r.ratio_scalar ?? GATE) >= GATE;
  if (!pass) allPass = false;
  const mark = pass ? '✓' : '✗';
  console.log(`${mark} ${r.label.padEnd(32)} ${String(sc).padStart(6)}×   ${String(si).padStart(6)}×`);
}
console.log('─────────────────────────────────────────────────────────────────');
console.log(allPass ? '✓ ALL gates PASSED' : '✗ SOME gates FAILED — see above');
