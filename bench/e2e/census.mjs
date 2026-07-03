/**
 * 5-step census pipeline benchmark: databonk vs Arquero.
 *
 * Measures the EXACT steps from the user's real-world census workload:
 *   1. load names  — fromArrow (names table: GEOID + name, plain utf8)
 *   2. load pops   — fromArrow (pops  table: GEOID + 72 numeric cols, plain utf8)
 *   3. join        — inner join on GEOID (unique key → 1:1 match)
 *   4. derive      — state = GEOID.slice(0, 11)
 *                    CP.3: col('GEOID').str.slice(0, 11) — dict-values path (fast)
 *                    CP.3: mapFn escape-hatch — also timed for comparison
 *   5. groupby sum — groupby(state).agg({<firstPopCol>: 'sum'})
 *
 * Usage:
 *   node bench/e2e/census.mjs           # print table to stdout
 *   node bench/e2e/census.mjs --json    # also write bench/baselines/census-v0.json
 *
 * Requires dist/ to be built (npm run build) — run in Docker via:
 *   docker run --rm -v /path/to/repo:/work -w /work dataframe-dev bash -lc \
 *     'npm ci && npm run build && node bench/e2e/census.mjs --json'
 */

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import * as aq from 'arquero';

import { generateCensusData } from './census-data.mjs';
import { DataFrame, fromArrow, runtimeFromExports, useRuntime, col } from '../../dist/index.js';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '../../wasm/dist');
const BASELINES_DIR = join(__dir, '../baselines');
const WARM = 3;
const MEASURE = 3;

// ─── Runtime init ─────────────────────────────────────────────────────────────

async function loadRuntime() {
  const buf = await readFile(join(WASM_DIR, 'simd.wasm'));
  const { instance } = await WebAssembly.instantiate(buf, {});
  return runtimeFromExports(instance.exports, true);
}

// ─── Timer ────────────────────────────────────────────────────────────────────

/**
 * Run `fn` `warm` times (discard), then `measure` times, return median ms.
 * Supports async and sync functions.
 */
async function timeMs(fn, warm, measure) {
  for (let i = 0; i < warm; i++) await fn();
  const ts = [];
  for (let i = 0; i < measure; i++) {
    const t0 = performance.now();
    await fn();
    ts.push(performance.now() - t0);
  }
  ts.sort((a, b) => a - b);
  return ts[Math.floor(ts.length / 2)];
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const writeJson = process.argv.includes('--json');

  // ── Step 0: init runtime (databonk only) ──────────────────────────────────
  const t0rt = performance.now();
  const rt = await loadRuntime();
  useRuntime(rt);
  const rtMs = performance.now() - t0rt;

  // ── Generate Arrow buffers (outside bench loop — same data for all runs) ──
  console.log('Generating synthetic census data (85,395 rows)…');
  const t0gen = performance.now();
  const { namesBuf, popsBuf, popColNames } = generateCensusData();
  console.log(`  Generated in ${(performance.now() - t0gen).toFixed(0)}ms`);
  console.log(`  names buf: ${(namesBuf.byteLength / 1024).toFixed(0)} KB`);
  console.log(`  pops  buf: ${(popsBuf.byteLength / 1024).toFixed(0)} KB`);
  console.log(`  pop columns: ${popColNames.length} (first: ${popColNames[0]})`);

  const firstPopCol = popColNames[0]; // 'pop_f0'

  // ─── databonk pipeline ─────────────────────────────────────────────────────

  // Step 1: fromArrow names
  const db_loadNames = await timeMs(() => {
    const df = fromArrow(namesBuf, rt);
    df.dispose();
  }, WARM, MEASURE);

  // Step 2: fromArrow pops
  const db_loadPops = await timeMs(() => {
    const df = fromArrow(popsBuf, rt);
    df.dispose();
  }, WARM, MEASURE);

  // Pre-load for steps 3-5 (not timed here)
  const dfNames = fromArrow(namesBuf, rt);
  const dfPops  = fromArrow(popsBuf, rt);

  // Step 3: inner join on GEOID
  const db_join = await timeMs(() => {
    const j = dfNames.join(dfPops, { on: 'GEOID', how: 'inner' });
    j.dispose();
  }, WARM, MEASURE);

  // Build joined frame for steps 4-5
  const dfJoined = dfNames.join(dfPops, { on: 'GEOID', how: 'inner' });

  // Step 4: derive state = GEOID.slice(0, 11)
  // CP.3: str.slice on dictionary values — applies slice to unique values only,
  //        then remaps indices (O(unique) + O(rows), no per-row string work).
  const db_derive = await timeMs(() => {
    const dfState = dfJoined.withColumn('state', col('GEOID').str.slice(0, 11));
    dfState.dispose();
  }, WARM, MEASURE);

  // CP.3: also time the escape-hatch (mapFn) path for comparison.
  const db_derive_escape = await timeMs(() => {
    const states = dfJoined.mapFn(row => row.GEOID.slice(0, 11));
    const dfState = dfJoined.withColumn('state', states, { dtype: 'utf8' });
    dfState.dispose();
  }, WARM, MEASURE);

  const dfWithState = dfJoined.withColumn('state', col('GEOID').str.slice(0, 11));

  // Step 5: groupby(state).sum(firstPopCol)
  const db_groupby = await timeMs(() => {
    const agg = {};
    agg[firstPopCol] = 'sum';
    const r = dfWithState.groupby('state').agg(agg);
    r.dispose();
  }, WARM, MEASURE);

  dfWithState.dispose();
  dfJoined.dispose();
  dfNames.dispose();
  dfPops.dispose();

  // ─── Arquero pipeline ──────────────────────────────────────────────────────

  // Step 1: aq.fromArrow names
  const aq_loadNames = await timeMs(() => {
    aq.fromArrow(namesBuf);
  }, WARM, MEASURE);

  // Step 2: aq.fromArrow pops
  const aq_loadPops = await timeMs(() => {
    aq.fromArrow(popsBuf);
  }, WARM, MEASURE);

  // Pre-load Arquero tables
  const aqNames = aq.fromArrow(namesBuf);
  const aqPops  = aq.fromArrow(popsBuf);

  // Step 3: join
  const aq_join = await timeMs(() => {
    aqNames.join(aqPops, 'GEOID');
  }, WARM, MEASURE);

  const aqJoined = aqNames.join(aqPops, 'GEOID');

  // Step 4: derive state = GEOID.slice(0, 11)
  const aq_derive = await timeMs(() => {
    aqJoined.derive({ state: d => aq.op.substring(d.GEOID, 0, 11) });
  }, WARM, MEASURE);

  const aqWithState = aqJoined.derive({ state: d => aq.op.substring(d.GEOID, 0, 11) });

  // Step 5: groupby(state).rollup({firstPopCol: sum})
  const aq_groupby = await timeMs(() => {
    const spec = {};
    spec[firstPopCol] = aq.op.sum(firstPopCol);
    aqWithState.groupby('state').rollup(spec);
  }, WARM, MEASURE);

  // ─── Results ──────────────────────────────────────────────────────────────

  const steps = [
    { name: 'init runtime',        db: rtMs,               aq: 0 },
    { name: 'load names',          db: db_loadNames,        aq: aq_loadNames },
    { name: 'load pops(72c)',      db: db_loadPops,         aq: aq_loadPops  },
    { name: 'join GEOID',          db: db_join,             aq: aq_join      },
    { name: 'derive state(str)',   db: db_derive,           aq: aq_derive    },
    { name: 'derive state(mapFn)', db: db_derive_escape,    aq: null         },
    { name: 'groupby sum',         db: db_groupby,          aq: aq_groupby   },
  ];

  // Totals use the str.slice path (excluding init runtime for fair comparison)
  const dbTotal = db_loadNames + db_loadPops + db_join + db_derive + db_groupby;
  const aqTotal = aq_loadNames + aq_loadPops + aq_join + aq_derive + aq_groupby;

  console.log(`\nCensus pipeline @ 85,395 rows (SIMD build, ${MEASURE} runs, median)\n`);
  const hdr = ['step'.padEnd(22), 'databonk(ms)'.padStart(12), 'arquero(ms)'.padStart(12), 'ratio(db/aq)'.padStart(13)];
  console.log(hdr.join(''));
  console.log('-'.repeat(61));

  for (const { name, db, aq: aqMs } of steps) {
    const ratio = (aqMs != null && aqMs > 0) ? (db / aqMs).toFixed(2) + '×' : 'n/a';
    console.log([
      name.padEnd(22),
      db.toFixed(1).padStart(12),
      ((aqMs != null && aqMs > 0) ? aqMs.toFixed(1) : 'n/a').padStart(12),
      ratio.padStart(13),
    ].join(''));
  }

  console.log('-'.repeat(61));
  console.log([
    'TOTAL (excl init)'.padEnd(22),
    dbTotal.toFixed(1).padStart(12),
    aqTotal.toFixed(1).padStart(12),
    (dbTotal / aqTotal).toFixed(2).padStart(12) + '×',
  ].join(''));
  console.log('');

  // ─── JSON output ──────────────────────────────────────────────────────────

  if (writeJson) {
    const baseline = {
      meta: {
        rows: 85_395,
        warm: WARM,
        measure: MEASURE,
        date: new Date().toISOString(),
        note: 'CP.3 — str.slice on dictionary values (census-perf workstream)',
      },
      steps: {},
    };
    for (const { name, db, aq: aqMs } of steps) {
      baseline.steps[name] = {
        databonk_ms: +db.toFixed(2),
        arquero_ms:  (aqMs != null && aqMs > 0) ? +aqMs.toFixed(2) : null,
        ratio:       (aqMs != null && aqMs > 0) ? +(db / aqMs).toFixed(3) : null,
      };
    }
    baseline.steps['TOTAL'] = {
      databonk_ms: +dbTotal.toFixed(2),
      arquero_ms:  +aqTotal.toFixed(2),
      ratio:       +(dbTotal / aqTotal).toFixed(3),
    };

    await mkdir(BASELINES_DIR, { recursive: true });
    const outPath = join(BASELINES_DIR, 'census-v0.json');
    await writeFile(outPath, JSON.stringify(baseline, null, 2));
    console.log(`Baseline written to ${outPath}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
