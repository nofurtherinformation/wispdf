/**
 * Benchmark harness — Phase 0.2
 *
 * Usage:
 *   node bench/run.mjs [--sizes=100k[,1m,10m]] [--baseline=<name>] [--op=<name>]
 *   npm run bench -- --sizes=100k,1m --baseline=typedarray
 *
 * Writes results to bench/baselines/{baseline}.json
 */

import { Bench } from 'tinybench';
import { execSync } from 'child_process';
import { writeFileSync, mkdirSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

import { makeNumericRaw, makeStringRaw } from './datasets.mjs';

// ── baseline modules ──────────────────────────────────────────────────────
import * as aoMod from './baselines/arrayobj.mjs';
import * as taMod from './baselines/typedarray.mjs';
import * as aqMod from './baselines/arquero.mjs';

const __dir = dirname(fileURLToPath(import.meta.url));

// ── CLI arg parsing ───────────────────────────────────────────────────────
const argv = process.argv.slice(2);

function argVal(name) {
  const prefix = `--${name}=`;
  const found = argv.find((a) => a.startsWith(prefix));
  return found ? found.slice(prefix.length) : null;
}

const sizeArg = argVal('sizes') ?? '100k,1m';
const baselineArg = argVal('baseline');
const opArg = argVal('op');

const SIZE_MAP = { '100k': 100_000, '1m': 1_000_000, '10m': 10_000_000 };
const requestedSizes = sizeArg.split(',').map((s) => {
  const n = SIZE_MAP[s.toLowerCase()];
  if (!n) throw new Error(`Unknown size: ${s}. Valid: 100k, 1m, 10m`);
  return n;
});

const ALL_BASELINES = ['arrayobj', 'typedarray', 'arquero', 'danfo'];
const activeBaselines = baselineArg ? [baselineArg] : ALL_BASELINES;

// ── baseline registry ─────────────────────────────────────────────────────
// danfo is loaded lazily to capture skip status
let danfoMod;
async function getDanfo() {
  if (!danfoMod) danfoMod = await import('./baselines/danfo.mjs');
  return danfoMod;
}

const MODULES = {
  arrayobj: aoMod,
  typedarray: taMod,
  arquero: aqMod,
};

// ── env ───────────────────────────────────────────────────────────────────
const nodeVersion = process.version;

// ── benchmark config ──────────────────────────────────────────────────────
const BENCH_TIME_MS = 2000; // min time per benchmark
const BENCH_WARMUP = 3;

// ── ops definition ────────────────────────────────────────────────────────
// Each entry: { name, dataset: 'numeric'|'string', fn: (mod, ds) => () => any }
// dataset = 'string' means use the string dataset (has col g); otherwise numeric.
const OPS = [
  {
    name: 'add_ab',
    dataset: 'numeric',
    fn: (mod, ds) => () => mod.opAdd(ds),
  },
  {
    name: 'filter_a_gt_0_5',
    dataset: 'numeric',
    fn: (mod, ds) => () => mod.opFilter(ds),
  },
  {
    name: 'sum_a',
    dataset: 'numeric',
    fn: (mod, ds) => () => mod.opSum(ds),
  },
  {
    name: 'groupby_g_sum_a',
    dataset: 'string',
    fn: (mod, ds) => () => mod.opGroupbySum(ds),
  },
  {
    name: 'pipeline_filter_groupby_sum',
    dataset: 'string',
    fn: (mod, ds) => () => mod.opPipeline(ds),
  },
];

const activeOps = opArg ? OPS.filter((o) => o.name === opArg) : OPS;

// ── result accumulator ────────────────────────────────────────────────────
// baseline -> results[]
const resultsByBaseline = {};
for (const bl of ALL_BASELINES) {
  resultsByBaseline[bl] = [];
}

// ── dataset cache (avoid regenerating per baseline) ──────────────────────
const rawNumericCache = new Map();
const rawStringCache = new Map();

function getRawNumeric(n) {
  if (!rawNumericCache.has(n)) rawNumericCache.set(n, makeNumericRaw(n));
  return rawNumericCache.get(n);
}
function getRawString(n) {
  if (!rawStringCache.has(n)) rawStringCache.set(n, makeStringRaw(n));
  return rawStringCache.get(n);
}

// ── main loop ─────────────────────────────────────────────────────────────
for (const baseline of activeBaselines) {
  // 10M rows only for typedarray
  const sizes =
    baseline !== 'typedarray'
      ? requestedSizes.filter((n) => n <= 1_000_000)
      : requestedSizes;

  console.log(`\n=== Baseline: ${baseline} (sizes: ${sizes.map((n) => n.toLocaleString()).join(', ')}) ===`);

  let mod;
  let skip = null;

  if (baseline === 'danfo') {
    mod = await getDanfo();
    skip = mod.SKIP;
  } else {
    mod = MODULES[baseline];
  }

  if (skip) {
    console.log(`  SKIPPED: ${skip.reason}`);
    resultsByBaseline[baseline].push({ status: 'skipped', reason: skip.reason });
    continue;
  }

  for (const n of sizes) {
    // Build dataset for this baseline + size
    const rawNum = getRawNumeric(n);
    const rawStr = n <= 1_000_000 ? getRawString(n) : null;

    const dsNumeric = mod.buildNumeric(rawNum);
    const dsString = rawStr ? mod.buildString(rawStr) : null;

    for (const opDef of activeOps) {
      // String ops only run at ≤ 1M rows
      if (opDef.dataset === 'string' && n > 1_000_000) continue;
      if (opDef.dataset === 'string' && !dsString) continue;

      const ds = opDef.dataset === 'string' ? dsString : dsNumeric;
      const benchFn = opDef.fn(mod, ds);

      process.stdout.write(`  ${opDef.name} @ ${(n / 1e3).toFixed(0)}K rows ... `);

      const bench = new Bench({ time: BENCH_TIME_MS, warmupIterations: BENCH_WARMUP });
      bench.add(opDef.name, benchFn);
      await bench.run();

      const task = bench.tasks[0];
      const { mean, sd } = task.result;

      // tinybench returns mean in ms
      const msMean = mean;
      const msStddev = sd;
      const opsPerSec = 1000 / msMean;

      console.log(`${msMean.toFixed(3)} ms ± ${msStddev.toFixed(3)} ms`);

      resultsByBaseline[baseline].push({
        op: opDef.name,
        rows: n,
        ops_per_sec: opsPerSec,
        ms_mean: msMean,
        ms_stddev: msStddev,
      });
    }
  }
}

// ── write JSON files ──────────────────────────────────────────────────────
const outDir = join(__dir, 'baselines');
mkdirSync(outDir, { recursive: true });

for (const baseline of activeBaselines) {
  const results = resultsByBaseline[baseline];
  if (!results.length) continue;

  const output = {
    baseline,
    env: { node: nodeVersion },
    results,
  };

  const outPath = join(outDir, `${baseline}.json`);
  writeFileSync(outPath, JSON.stringify(output, null, 2) + '\n');
  console.log(`\nWrote ${outPath}`);
}

console.log('\nDone.');
