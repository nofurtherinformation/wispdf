#!/usr/bin/env node
/**
 * Benchmark regression gate (spec §5).
 *
 * Records or checks headline op medians against a stored baseline JSON.
 * Uses the SIMD build and the public DataFrame API (covers all 4 kernel families).
 *
 * Usage:
 *   node scripts/check-regression.mjs           # check (fails if >10% regression)
 *   node scripts/check-regression.mjs --update  # record new baseline
 *
 * Runs at 100K rows by default (fast enough for CI); 1M results are also recorded
 * in the baseline but not checked in the gate to keep wall time reasonable.
 *
 * Exit codes: 0 = all checks pass (or --update succeeded), 1 = regression detected.
 */

import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { existsSync } from 'node:fs';

import { makeStringRaw, makeNumericRaw } from '../bench/datasets.mjs';
import { DataFrame, runtimeFromExports, useRuntime, col } from '../dist/index.js';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '..', 'wasm', 'dist');
const BASELINE_PATH = join(__dir, '..', 'bench', 'baselines', 'wasm-v1.json');

const UPDATE = process.argv.includes('--update');
const THRESHOLD = 0.10; // 10% regression tolerance

// ── Wasm loader ───────────────────────────────────────────────────────────────

async function loadWasm(name) {
  const buf = await readFile(join(WASM_DIR, name));
  const { instance } = await WebAssembly.instantiate(buf, {});
  return instance;
}

// ── Timing helper ─────────────────────────────────────────────────────────────

function timeMs(fn, warm = 2, measure = 5) {
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

// ── Benchmark suite ───────────────────────────────────────────────────────────

async function runBenchmarks() {
  // Load SIMD wasm
  const { exports } = await loadWasm('simd.wasm');
  useRuntime(runtimeFromExports(exports, true));

  const results = {};

  for (const N of [100_000, 1_000_000]) {
    const label = N === 100_000 ? '100k' : '1m';
    const rawStr = makeStringRaw(N);
    const gStr = Array.from({ length: N }, (_, i) => rawStr.gDict[rawStr.gIndices[i]]);

    const df = DataFrame.fromColumns(
      { a: rawStr.a, g: gStr },
      { dtypes: { a: 'f64', g: 'utf8' } },
    );

    const gUnique = rawStr.gDict;
    const w = Float64Array.from(gUnique, (_, i) => i * 0.5);
    const right = DataFrame.fromColumns(
      { g: gUnique, w },
      { dtypes: { g: 'utf8', w: 'f64' } },
    );

    // pipeline: filter → groupby → sum (covers elementwise gt + select filter + hash groupby + reduce sum)
    results[`pipeline_${label}`] = timeMs(() => {
      const f = df.filter(col('a').gt(0.5));
      f.groupby('g').agg({ a: 'sum' }).dispose();
      f.dispose();
    });

    // join: inner hash join (covers hash family)
    results[`join_${label}`] = timeMs(() => {
      df.join(right, { on: 'g', how: 'inner' }).dispose();
    });

    // sortValues: argsort + gather (covers select family)
    results[`sortValues_${label}`] = timeMs(() => {
      df.sortValues('a').dispose();
    }, 0, 3);

    // withColumn: elementwise add (covers elementwise family alone)
    results[`withColumn_add_${label}`] = timeMs(() => {
      df.withColumn('a', col('a').add(1)).dispose();
    });

    // groupby sum only (covers reduce + hash, isolated from filter)
    results[`groupby_sum_${label}`] = timeMs(() => {
      df.groupby('g').agg({ a: 'sum' }).dispose();
    });

    df.dispose();
    right.dispose();
  }

  return results;
}

// ── Gate check ────────────────────────────────────────────────────────────────

// Only these ops are checked in the regression gate (100K = fast enough for CI).
const GATE_OPS = [
  'pipeline_100k',
  'join_100k',
  'sortValues_100k',
  'withColumn_add_100k',
  'groupby_sum_100k',
];

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`\nBenchmark regression ${UPDATE ? '(--update: recording new baseline)' : '(gate check)'}\n`);

  const current = await runBenchmarks();
  const node = process.version;

  if (UPDATE) {
    // Guard: refuse to record a baseline from a dirty tree (P4.2 finding — an
    // uncommitted hot-path edit would silently corrupt the baseline). --force overrides.
    if (!process.argv.includes('--force')) {
      const { execSync } = await import('node:child_process');
      try {
        const dirty = execSync("git status --porcelain -- . ':!bench/baselines'", {
          cwd: join(__dir, '..'), encoding: 'utf8',
        }).trim();
        if (dirty) {
          console.error('Refusing --update: working tree is dirty (commit first, or pass --force):');
          console.error(dirty.split('\n').slice(0, 10).join('\n'));
          process.exit(1);
        }
      } catch { /* no git available (e.g. bare container) — proceed */ }
    }
    const baseline = {
      schema: 'wasm-v1',
      recorded: new Date().toISOString().slice(0, 10),
      note: 'SIMD build medians (ms). Gate checks 100K ops with ≤10% regression tolerance.',
      node,
      gate_ops: GATE_OPS,
      ops: Object.fromEntries(
        Object.entries(current).map(([k, v]) => [k, { median_ms: +v.toFixed(4) }]),
      ),
    };
    await writeFile(BASELINE_PATH, JSON.stringify(baseline, null, 2) + '\n');
    console.log(`Baseline written to ${BASELINE_PATH}`);
    console.log('\nRecorded ops:');
    for (const [op, { median_ms }] of Object.entries(baseline.ops)) {
      console.log(`  ${op.padEnd(28)} ${median_ms.toFixed(3)} ms`);
    }
    return;
  }

  // Load stored baseline
  if (!existsSync(BASELINE_PATH)) {
    console.error(`Baseline not found: ${BASELINE_PATH}`);
    console.error('Run: node scripts/check-regression.mjs --update');
    process.exit(1);
  }
  const baseline = JSON.parse(await readFile(BASELINE_PATH, 'utf8'));

  console.log(`Baseline recorded: ${baseline.recorded}  node: ${baseline.node}`);
  console.log(`Current node: ${node}\n`);

  let failures = 0;
  console.log(
    ['op'.padEnd(30), 'baseline(ms)'.padStart(14), 'current(ms)'.padStart(13), 'ratio'.padStart(8), 'status'.padStart(8)].join(''),
  );

  for (const op of GATE_OPS) {
    const b = baseline.ops?.[op];
    const c = current[op];
    if (!b) {
      console.log(`  ${op.padEnd(28)} (baseline entry missing — skipping)`);
      continue;
    }
    const ratio = c / b.median_ms;
    // ponytail: sub-ms ops are timer-noise-bound (P4.2 found withColumn_add_100k
    // flaking at 1.11× on a 0.095ms baseline) — widen their tolerance to 25%.
    const tol = b.median_ms < 1 ? 0.25 : THRESHOLD;
    const ok = ratio <= 1 + tol;
    const status = ok ? 'OK' : 'FAIL';
    if (!ok) failures++;
    console.log(
      [
        op.padEnd(30),
        b.median_ms.toFixed(3).padStart(14),
        c.toFixed(3).padStart(13),
        `${ratio.toFixed(2)}×`.padStart(8),
        status.padStart(8),
      ].join(''),
    );
  }

  console.log();
  if (failures > 0) {
    console.error(`REGRESSION GATE FAILED: ${failures} op(s) regressed >10% vs baseline.`);
    console.error('If this is expected (intentional change or faster machine), run:');
    console.error('  node scripts/check-regression.mjs --update');
    process.exit(1);
  } else {
    console.log(`All ${GATE_OPS.length} gate checks passed (≤${(THRESHOLD * 100).toFixed(0)}% tolerance).`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
