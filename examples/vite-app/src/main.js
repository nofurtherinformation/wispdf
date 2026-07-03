/**
 * databonk Vite demo — 100K-row pipeline with table preview.
 *
 * Wasm loading strategy: the postinstall script copies scalar.wasm + simd.wasm
 * to public/ so Vite serves them at / regardless of build target. For production
 * apps, prefer the `new URL('../node_modules/databonk/dist/simd.wasm', import.meta.url)`
 * pattern so Vite hashes and copies the asset automatically (see docs/bundlers.md).
 */

import {
  runtimeFromExports,
  useRuntime,
  detectSimd,
  DataFrame,
  col,
  scope,
} from 'databonk';

const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const logEl    = document.getElementById('log');

function log(msg) {
  logEl.innerHTML += `<pre>${msg}</pre>`;
  console.log(msg);
}

function setStatus(ok, msg) {
  statusEl.innerHTML = `<span class="status ${ok ? 'ok' : 'err'}">${msg}</span>`;
}

async function main() {
  // ── 1. Load wasm ────────────────────────────────────────────────────────
  const simd = detectSimd();
  const wasmFile = simd ? 'simd.wasm' : 'scalar.wasm';
  log(`loading ${wasmFile} (simd=${simd})`);

  // Wasm binaries are in public/ (copied by scripts/copy-wasm.mjs postinstall).
  // This absolute-path fetch works in both dev-server and built output.
  const resp = await fetch(`/${wasmFile}`);
  if (!resp.ok) throw new Error(`fetch /${wasmFile} failed: ${resp.status}`);
  const bytes = await resp.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  useRuntime(runtimeFromExports(instance.exports, simd));
  log(`wasm instantiated (${(bytes.byteLength / 1024).toFixed(1)} KB uncompressed)`);

  // ── 2. Build 100K-row dataset ─────────────────────────────────────────
  const N = 100_000;
  const a = new Float64Array(N);
  const gLabels = ['alpha', 'beta', 'gamma', 'delta'];
  const g = new Array(N);
  for (let i = 0; i < N; i++) {
    a[i] = Math.random();
    g[i] = gLabels[i % gLabels.length];
  }

  const df = DataFrame.fromColumns({ a, g }, { dtypes: { a: 'f64', g: 'utf8' } });
  log(`DataFrame: ${df.length.toLocaleString()} rows × ${df.columns.length} columns`);

  // ── 3. Pipeline: filter → groupby → agg ──────────────────────────────
  const t0 = performance.now();
  const result = scope(() => {
    const filtered = df.filter(col('a').gt(0.5));
    const grouped  = filtered.groupby(['g']).agg({ a: ['sum', 'mean', 'count'] });
    return grouped.toRecords();
  });
  const elapsed = performance.now() - t0;

  log(`pipeline completed in ${elapsed.toFixed(2)} ms (${N.toLocaleString()} rows)`);

  // ── 4. Table preview ──────────────────────────────────────────────────
  const cols = Object.keys(result[0] ?? {});
  const thead = `<tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>`;
  const tbody = result
    .map(r => `<tr>${cols.map(c => {
      const v = r[c];
      return `<td>${typeof v === 'number' ? v.toFixed(4) : v}</td>`;
    }).join('')}</tr>`)
    .join('');
  resultEl.innerHTML = `<table><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;

  df.dispose();
  setStatus(true, `PASS — pipeline ${elapsed.toFixed(1)} ms @ 100K rows (${simd ? 'SIMD' : 'scalar'})`);
}

main().catch(err => {
  console.error(err);
  setStatus(false, `ERROR: ${err.message}`);
  log(`ERROR:\n${err.stack}`);
});
