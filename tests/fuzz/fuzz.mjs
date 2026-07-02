#!/usr/bin/env node
/**
 * Long-running fuzz soak — 1000 random pipelines against the public DataFrame API.
 *
 * Usage: npm run fuzz  (or node tests/fuzz/fuzz.mjs [--runs=N])
 *
 * Imports from dist/index.js (built bundle); run `npm run build` first.
 * Uses fast-check's fc.assert() so failures shrink to a minimal repro.
 */

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import fc from 'fast-check';

import { DataFrame, runtimeFromExports, useRuntime, col, lit } from '../../dist/index.js';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '..', '..', 'wasm', 'dist');

// ── CLI ───────────────────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
function argVal(name) {
  const found = argv.find((a) => a.startsWith(`--${name}=`));
  return found ? found.slice(name.length + 3) : null;
}
const NUM_RUNS = Number(argVal('runs') ?? 1000);
const BUILD = argVal('build') ?? 'scalar'; // 'scalar' | 'simd'

// ── Runtime load ──────────────────────────────────────────────────────────────
async function loadRuntime() {
  const wasmFile = BUILD === 'simd' ? 'simd.wasm' : 'scalar.wasm';
  const buf = await readFile(join(WASM_DIR, wasmFile));
  const { instance } = await WebAssembly.instantiate(buf, {});
  useRuntime(runtimeFromExports(instance.exports, BUILD === 'simd'));
  console.log(`Loaded ${wasmFile}`);
}

// ── Oracle ────────────────────────────────────────────────────────────────────

function makeOracleFrame(a, b, g, flag) {
  return Array.from({ length: a.length }, (_, i) => ({
    a: a[i] ?? null,
    b: b[i] ?? null,
    g: g[i] ?? null,
    flag: flag[i] ?? null,
  }));
}

function oracleFilterGt(frame, col, val) {
  return frame.filter((r) => {
    const v = r[col];
    return typeof v === 'number' && !isNaN(v) && v > val;
  });
}
function oracleFilterLt(frame, col, val) {
  return frame.filter((r) => {
    const v = r[col];
    return typeof v === 'number' && !isNaN(v) && v < val;
  });
}
function oracleFilterEqStr(frame, col, val) {
  return frame.filter((r) => r[col] === val);
}
function oracleWithColumnMul(frame, name, val) {
  return frame.map((r) => ({ ...r, [name]: r[name] === null ? null : r[name] * val }));
}
function oracleWithColumnAdd(frame, name, val) {
  return frame.map((r) => ({ ...r, [name]: r[name] === null ? null : r[name] + val }));
}
function oracleWithColumnI32Mul(frame, name, val) {
  return frame.map((r) => ({ ...r, [name]: r[name] === null ? null : (r[name] * val) | 0 }));
}
function oracleWithColumnI32Add(frame, name, val) {
  return frame.map((r) => ({ ...r, [name]: r[name] === null ? null : (r[name] + val) | 0 }));
}
function oracleSortAsc(frame, col) {
  return frame.slice().sort((ra, rb) => {
    const a = ra[col], b = rb[col];
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;
    if (typeof a === 'number' && isNaN(a) && typeof b === 'number' && isNaN(b)) return 0;
    if (typeof a === 'number' && isNaN(a)) return 1;
    if (typeof b === 'number' && isNaN(b)) return -1;
    return a < b ? -1 : a > b ? 1 : 0;
  });
}
function oracleSortDesc(frame, col) {
  return frame.slice().sort((ra, rb) => {
    const a = ra[col], b = rb[col];
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;
    if (typeof a === 'number' && isNaN(a) && typeof b === 'number' && isNaN(b)) return 0;
    if (typeof a === 'number' && isNaN(a)) return 1;
    if (typeof b === 'number' && isNaN(b)) return -1;
    return a < b ? 1 : a > b ? -1 : 0;
  });
}
function oracleGroupbyAgg(frame, key, aggCol, fn) {
  const groups = new Map();
  for (const r of frame) {
    const k = r[key] ?? null;
    const v = r[aggCol];
    const kStr = JSON.stringify(k);
    if (!groups.has(kStr)) groups.set(kStr, { key: k, vals: [] });
    if (v !== null && (typeof v !== 'number' || !isNaN(v))) groups.get(kStr).vals.push(v);
  }
  return [...groups.values()].map(({ key: k, vals }) => {
    let agg;
    switch (fn) {
      case 'sum': agg = vals.reduce((s, x) => s + x, 0); break;
      case 'mean': agg = vals.length > 0 ? vals.reduce((s, x) => s + x, 0) / vals.length : null; break;
      case 'count': agg = vals.length; break;
      case 'min': agg = vals.length > 0 ? Math.min(...vals) : null; break;
      case 'max': agg = vals.length > 0 ? Math.max(...vals) : null; break;
    }
    return { [key]: k, [aggCol]: agg };
  });
}

function cellEq(a, b) {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  if (typeof a === 'number' && typeof b === 'number') {
    if (isNaN(a) && isNaN(b)) return true;
    return a === b;
  }
  return a === b;
}

function compareFrames(libCols, oracle, cols) {
  const n = oracle.length;
  for (const c of cols) {
    const libCol = libCols[c];
    if (!libCol) return `col '${c}' missing`;
    if (libCol.length !== n) return `col '${c}' length: lib=${libCol.length} oracle=${n}`;
    for (let i = 0; i < n; i++) {
      const lo = libCol[i] ?? null;
      const oo = oracle[i]?.[c] ?? null;
      if (!cellEq(lo, oo)) return `row ${i} col '${c}': lib=${JSON.stringify(lo)} oracle=${JSON.stringify(oo)}`;
    }
  }
  return null;
}

// ── Arbitraries ───────────────────────────────────────────────────────────────
const STR_VALS = ['alpha', 'beta', 'gamma', 'delta'];

const f64Cell = fc.option(
  fc.oneof(
    fc.double({ min: -100, max: 100, noNaN: false, noDefaultInfinity: false }),
    fc.constantFrom(Infinity, -Infinity),
  ),
  { nil: null, freq: 8 },
);
const i32Cell = fc.option(fc.integer({ min: -30, max: 30 }), { nil: null, freq: 8 });
const utf8Cell = fc.option(fc.constantFrom(...STR_VALS), { nil: null, freq: 8 });
const boolCell = fc.option(fc.boolean(), { nil: null, freq: 8 });

const frameArb = fc.nat({ max: 50 }).chain((n) =>
  fc.record({
    a: fc.array(f64Cell, { minLength: n, maxLength: n }),
    b: fc.array(i32Cell, { minLength: n, maxLength: n }),
    g: fc.array(utf8Cell, { minLength: n, maxLength: n }),
    flag: fc.array(boolCell, { minLength: n, maxLength: n }),
  }),
);

const stepArb = fc.oneof(
  fc.record({ kind: fc.constant('filterAGt'), val: fc.integer({ min: -50, max: 50 }) }),
  fc.record({ kind: fc.constant('filterALt'), val: fc.integer({ min: -50, max: 50 }) }),
  fc.record({ kind: fc.constant('filterBGt'), val: fc.integer({ min: -25, max: 25 }) }),
  fc.record({ kind: fc.constant('filterBLt'), val: fc.integer({ min: -25, max: 25 }) }),
  fc.record({ kind: fc.constant('filterGEq'), val: fc.constantFrom(...STR_VALS) }),
  fc.record({ kind: fc.constant('mulA'), val: fc.integer({ min: 1, max: 4 }) }),
  fc.record({ kind: fc.constant('addA'), val: fc.double({ min: -10, max: 10, noNaN: true, noDefaultInfinity: true }) }),
  fc.record({ kind: fc.constant('mulB'), val: fc.integer({ min: 1, max: 4 }) }),
  fc.record({ kind: fc.constant('addB'), val: fc.integer({ min: -10, max: 10 }) }),
  fc.record({ kind: fc.constant('sortAsc'), col: fc.constantFrom('b', 'g') }),
  fc.record({ kind: fc.constant('sortDesc'), col: fc.constantFrom('b', 'g') }),
  fc.record({ kind: fc.constant('head'), n: fc.nat({ max: 30 }) }),
  fc.record({ kind: fc.constant('slice'), i: fc.nat({ max: 25 }), j: fc.nat({ max: 50 }) }),
);

const pipelineArb = fc.array(stepArb, { minLength: 0, maxLength: 6 });
const aggArb = fc.constantFrom('sum', 'mean', 'count', 'min', 'max');

// ── Apply steps ───────────────────────────────────────────────────────────────

function applyLibStep(df, step) {
  switch (step.kind) {
    case 'filterAGt': return df.filter(col('a').gt(lit(step.val)));
    case 'filterALt': return df.filter(col('a').lt(lit(step.val)));
    case 'filterBGt': return df.filter(col('b').gt(lit(step.val)));
    case 'filterBLt': return df.filter(col('b').lt(lit(step.val)));
    case 'filterGEq': return df.filter(col('g').eq(lit(step.val)));
    case 'mulA': return df.withColumn('a', col('a').mul(lit(step.val)));
    case 'addA': return df.withColumn('a', col('a').add(lit(step.val)));
    case 'mulB': return df.withColumn('b', col('b').mul(lit(step.val)));
    case 'addB': return df.withColumn('b', col('b').add(lit(step.val)));
    case 'sortAsc': return df.sortValues(step.col, { descending: false });
    case 'sortDesc': return df.sortValues(step.col, { descending: true });
    case 'head': return df.head(step.n);
    case 'slice': return df.slice(step.i, step.j);
    default: throw new Error(`unknown step ${step.kind}`);
  }
}

function applyOracleStep(frame, step) {
  switch (step.kind) {
    case 'filterAGt': return oracleFilterGt(frame, 'a', step.val);
    case 'filterALt': return oracleFilterLt(frame, 'a', step.val);
    case 'filterBGt': return oracleFilterGt(frame, 'b', step.val);
    case 'filterBLt': return oracleFilterLt(frame, 'b', step.val);
    case 'filterGEq': return oracleFilterEqStr(frame, 'g', step.val);
    case 'mulA': return oracleWithColumnMul(frame, 'a', step.val);
    case 'addA': return oracleWithColumnAdd(frame, 'a', step.val);
    case 'mulB': return oracleWithColumnI32Mul(frame, 'b', step.val);
    case 'addB': return oracleWithColumnI32Add(frame, 'b', step.val);
    case 'sortAsc': return oracleSortAsc(frame, step.col);
    case 'sortDesc': return oracleSortDesc(frame, step.col);
    case 'head': return oracleHead(frame, step.n);
    case 'slice': return oracleSlice(frame, step.i, step.j);
    default: throw new Error(`unknown step ${step.kind}`);
  }
}
function oracleHead(frame, n) { return frame.slice(0, n); }
function oracleSlice(frame, i, j) { return frame.slice(i, j); }

// ── Main soak ─────────────────────────────────────────────────────────────────

async function main() {
  await loadRuntime();
  console.log(`\nFuzz soak: ${NUM_RUNS} random pipelines on ${BUILD} build\n`);

  let passed = 0;
  let failed = 0;

  // Pipeline test
  console.log('--- schema-preserving pipelines ---');
  try {
    fc.assert(
      fc.property(frameArb, pipelineArb, ({ a, b, g, flag }, steps) => {
        const init = DataFrame.fromColumns(
          { a, b, g, flag },
          { dtypes: { a: 'f64', b: 'i32', g: 'utf8', flag: 'bool' } },
        );
        const owned = [init];
        try {
          let current = init;
          let oracle = makeOracleFrame(a, b, g, flag);
          for (const step of steps) {
            const next = applyLibStep(current, step);
            owned.push(next);
            oracle = applyOracleStep(oracle, step);
            current = next;
          }
          const libCols = current.toColumns();
          const err = compareFrames(libCols, oracle, ['a', 'b', 'g', 'flag']);
          if (err) throw new Error(err);
        } finally {
          for (const df of owned) df.dispose();
        }
      }),
      { numRuns: NUM_RUNS },
    );
    console.log(`PASS (${NUM_RUNS} runs)`);
    passed++;
  } catch (e) {
    console.error('FAIL:', e.message);
    failed++;
  }

  // Groupby test
  console.log('\n--- terminal groupby-agg ---');
  try {
    fc.assert(
      fc.property(frameArb, pipelineArb, aggArb, ({ a, b, g, flag }, steps, aggFn) => {
        const init = DataFrame.fromColumns(
          { a, b, g, flag },
          { dtypes: { a: 'f64', b: 'i32', g: 'utf8', flag: 'bool' } },
        );
        const owned = [init];
        try {
          let current = init;
          let oracle = makeOracleFrame(a, b, g, flag);
          for (const step of steps) {
            const next = applyLibStep(current, step);
            owned.push(next);
            oracle = applyOracleStep(oracle, step);
            current = next;
          }
          const aggResult = current.groupby('g').agg({ a: aggFn });
          owned.push(aggResult);
          const sorted = aggResult.sortValues('g', { descending: false });
          owned.push(sorted);
          const oracleAgg = oracleGroupbyAgg(oracle, 'g', 'a', aggFn);

          const libCols = sorted.toColumns();
          const libMap = new Map();
          const libG = libCols['g'] ?? [];
          const libA = libCols['a'] ?? [];
          for (let i = 0; i < libG.length; i++) libMap.set(libG[i] ?? null, libA[i] ?? null);

          if (libG.length !== oracleAgg.length) {
            throw new Error(`group count: lib=${libG.length} oracle=${oracleAgg.length}`);
          }
          for (const row of oracleAgg) {
            const oKey = row['g'] ?? null;
            const oVal = row['a'] ?? null;
            if (!libMap.has(oKey)) throw new Error(`key ${JSON.stringify(oKey)} missing`);
            const lVal = libMap.get(oKey) ?? null;
            if (aggFn === 'mean' || aggFn === 'sum') {
              if (oVal === null && lVal !== null) throw new Error(`${aggFn}[${oKey}]: lib=${lVal} oracle=null`);
              if (oVal !== null && lVal === null) throw new Error(`${aggFn}[${oKey}]: lib=null oracle=${oVal}`);
              if (oVal !== null && lVal !== null) {
                const diff = Math.abs(lVal - oVal) / Math.max(Math.abs(oVal), 1);
                if (diff > 1e-9) throw new Error(`${aggFn}[${oKey}]: lib=${lVal} oracle=${oVal}`);
              }
            } else {
              if (!cellEq(lVal, oVal)) throw new Error(`${aggFn}[${oKey}]: lib=${lVal} oracle=${oVal}`);
            }
          }
        } finally {
          for (const df of owned) df.dispose();
        }
      }),
      { numRuns: NUM_RUNS },
    );
    console.log(`PASS (${NUM_RUNS} runs)`);
    passed++;
  } catch (e) {
    console.error('FAIL:', e.message);
    failed++;
  }

  console.log(`\n${passed} suites passed, ${failed} failed.`);
  if (failed > 0) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
