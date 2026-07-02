// Independent correctness check for the Rust spike kernels (ADR-007 LEAD verification).
// Verifies add_f64, sum_f64_null (Arrow LSB validity), cmp_gt_f64_mask (LSB bitmask)
// on a deterministic 16-element fixture for BOTH scalar and simd builds.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DIST = join(__dirname, 'dist');

async function load(name) {
  const bytes = readFileSync(join(DIST, `${name}.wasm`));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

function check(name, cond, extra = '') {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}${extra ? '  ' + extra : ''}`);
  return cond;
}

async function verify(name) {
  console.log(`--- ${name} ---`);
  const ex = await load(name);
  const mem = ex.memory;
  const N = 16;
  const aPtr = ex.alloc(N * 8);
  const bPtr = ex.alloc(N * 8);
  const outPtr = ex.alloc(N * 8);
  const vPtr = ex.alloc(2);
  const mPtr = ex.alloc(2);

  const a = new Float64Array(mem.buffer, aPtr, N);
  const b = new Float64Array(mem.buffer, bPtr, N);
  const v = new Uint8Array(mem.buffer, vPtr, 2);
  for (let i = 0; i < N; i++) { a[i] = i + 1; b[i] = i * 2; }
  v[0] = 0x55; v[1] = 0x55; // valid at even indices 0,2,...,14

  let ok = true;

  ex.add_f64(aPtr, bPtr, outPtr, N);
  const out = new Float64Array(mem.buffer, outPtr, N);
  let addOk = true;
  for (let i = 0; i < N; i++) if (Math.abs(out[i] - ((i + 1) + i * 2)) > 1e-12) addOk = false;
  ok &= check('add_f64', addOk, `out[0..3]=${[...out.slice(0,4)]}`);

  // valid values = a at even idx = 1,3,5,7,9,11,13,15 -> sum 64
  const sum = ex.sum_f64_null(aPtr, vPtr, N);
  ok &= check('sum_f64_null', Math.abs(sum - 64) < 1e-10, `got=${sum} exp=64`);

  // scalar=8: a[i]=i+1; >8 for i>=8 -> byte0=0x00, byte1=0xFF
  ex.cmp_gt_f64_mask(aPtr, 8, mPtr, N);
  const m = new Uint8Array(mem.buffer, mPtr, 2);
  ok &= check('cmp_gt_f64_mask', m[0] === 0x00 && m[1] === 0xFF, `got=[${m[0]},${m[1]}] exp=[0,255]`);

  return !!ok;
}

const s = await verify('scalar');
const d = await verify('simd');
console.log(s && d ? '\nALL RUST KERNELS CORRECT' : '\nRUST KERNEL FAILURE');
process.exit(s && d ? 0 : 1);
