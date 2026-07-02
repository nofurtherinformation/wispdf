/**
 * End-to-end benchmark (spec §5 gate): filter(a>0.5) → groupby(g) → sum(a) at 1M rows,
 * this library vs Arquero live. Reports the ratio (Arquero / us; ≥1× passes the gate).
 * Also reports sortValues and join vs Arquero "for the record".
 *
 *   node bench/e2e/pipeline.mjs
 *
 * Imports the built bundle (`dist/index.js`) — run `npm run build` first — and loads the
 * SIMD wasm from `wasm/dist` directly through the public `runtimeFromExports`/`useRuntime`.
 */

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import * as aq from 'arquero';

import { makeStringRaw } from '../datasets.mjs';
import { DataFrame, runtimeFromExports, useRuntime, col } from '../../dist/index.js';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '../../wasm/dist');
const N = 1_000_000;

async function loadRuntime() {
  const buf = await readFile(join(WASM_DIR, 'simd.wasm'));
  const { instance } = await WebAssembly.instantiate(buf, {});
  useRuntime(runtimeFromExports(instance.exports, true));
}

/** Median wall time (ms) of `fn`, after `warm` warmup runs, over `measure` runs. */
function timeMs(fn, warm, measure) {
  for (let i = 0; i < warm; i++) fn();
  const ts = [];
  for (let i = 0; i < measure; i++) {
    const t0 = performance.now();
    fn();
    ts.push(performance.now() - t0);
  }
  ts.sort((a, b) => a - b);
  return ts[Math.floor(ts.length / 2)];
}

async function main() {
  await loadRuntime();
  const raw = makeStringRaw(N);
  const gStr = new Array(N);
  for (let i = 0; i < N; i++) gStr[i] = raw.gDict[raw.gIndices[i]];

  const df = DataFrame.fromColumns({ a: raw.a, g: gStr }, { dtypes: { a: 'f64', g: 'utf8' } });
  const gUnique = raw.gDict;
  const w = Float64Array.from(gUnique, (_, i) => i * 0.5);
  const right = DataFrame.fromColumns({ g: gUnique, w }, { dtypes: { g: 'utf8', w: 'f64' } });

  const aTable = aq.table({ a: Array.from(raw.a), g: gStr });
  const rightTable = aq.table({ g: gUnique, w: Array.from(w) });

  const rows = [
    [
      'pipeline (filter→groupby→sum) [GATE]',
      timeMs(() => {
        const f = df.filter(col('a').gt(0.5));
        f.groupby('g').agg({ a: 'sum' }).dispose();
        f.dispose();
      }, 3, 7),
      timeMs(() => aTable.filter((d) => d.a > 0.5).groupby('g').rollup({ a: aq.op.sum('a') }).objects(), 3, 7),
    ],
    [
      'join (inner)',
      timeMs(() => df.join(right, { on: 'g', how: 'inner' }).dispose(), 2, 5),
      timeMs(() => aTable.join(rightTable, 'g').objects(), 2, 5),
    ],
    [
      'sortValues',
      timeMs(() => df.sortValues('a').dispose(), 0, 1),
      timeMs(() => aTable.orderby('a').objects(), 1, 3),
    ],
  ];

  console.log(`\nE2E @ ${N.toLocaleString()} rows (SIMD build)\n`);
  console.log(['op'.padEnd(42), 'ours(ms)'.padStart(10), 'arquero(ms)'.padStart(12), 'ratio(×aq)'.padStart(12)].join(''));
  for (const [name, ours, arq] of rows) {
    console.log([name.padEnd(42), ours.toFixed(2).padStart(10), arq.toFixed(2).padStart(12), `${(arq / ours).toFixed(2)}×`.padStart(12)].join(''));
  }
  const gate = rows[0][2] / rows[0][1];
  console.log(`\nGATE (pipeline ≥ 1× Arquero): ${gate >= 1 ? 'PASS' : 'FAIL'} (${gate.toFixed(2)}×)\n`);

  df.dispose();
  right.dispose();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
