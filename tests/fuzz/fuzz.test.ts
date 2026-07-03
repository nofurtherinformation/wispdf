/**
 * Fuzz / property tests over the public DataFrame API.
 *
 * Uses fast-check to generate random frames (f64+nulls+NaN, i32+nulls, utf8+nulls, bool+nulls)
 * and random pipelines (chained filter/withColumn/sortValues/head/slice/groupby-agg), then
 * checks the library output against a naive JS oracle.  Shrinking is built into fast-check:
 * a failing case is automatically minimised before reporting.
 *
 * Gate (short) run: FUZZ_RUNS env var (default 60).
 * Long soak: npm run fuzz (standalone node script, 1000 pipelines).
 */
import { describe, it, expect, beforeAll } from 'vitest';
import fc from 'fast-check';

import { loadRuntimeForTest, makeDF } from '../frame/helper.js';
import { col, lit } from '../../src/expr/ast.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import type { DataFrame } from '../../src/frame/dataframe.js';
import type { Cell } from '../../src/memory/column.js';

import {
  makeOracleFrame,
  oracleFilterGt, oracleFilterLt, oracleFilterEqStr,
  oracleWithColumnMul, oracleWithColumnAdd,
  oracleWithColumnI32Mul, oracleWithColumnI32Add,
  oracleSortAsc, oracleSortDesc,
  oracleHead, oracleSlice, oracleSelect,
  oracleGroupbyAgg,
  compareFrames, cellEq,
  type OFrame,
} from './oracle.js';

// ── Test config ───────────────────────────────────────────────────────────────

const NUM_RUNS = Number(process.env['FUZZ_RUNS'] ?? 60);
const STR_VALS = ['alpha', 'beta', 'gamma', 'delta'];

// ── Runtime ───────────────────────────────────────────────────────────────────

let rt: DfRuntime;
beforeAll(async () => { rt = await loadRuntimeForTest(); });

// ── Arbitraries ───────────────────────────────────────────────────────────────

/** A nullable f64 value (includes NaN and ±Infinity). */
const f64Cell = fc.option(
  fc.oneof(
    fc.double({ min: -100, max: 100, noNaN: false, noDefaultInfinity: false }),
    fc.constantFrom(Infinity, -Infinity),
  ),
  { nil: null, freq: 8 },
);

/** A nullable i32 value. */
const i32Cell = fc.option(fc.integer({ min: -30, max: 30 }), { nil: null, freq: 8 });

/** A nullable utf8 value from a small dictionary. */
const utf8Cell = fc.option(fc.constantFrom(...STR_VALS), { nil: null, freq: 8 });

/** A nullable bool value. */
const boolCell = fc.option(fc.boolean(), { nil: null, freq: 8 });

/** A random frame with fixed schema {a:f64, b:i32, g:utf8, flag:bool}. */
const frameArb = fc.nat({ max: 50 }).chain((n) =>
  fc.record({
    a: fc.array(f64Cell, { minLength: n, maxLength: n }),
    b: fc.array(i32Cell, { minLength: n, maxLength: n }),
    g: fc.array(utf8Cell, { minLength: n, maxLength: n }),
    flag: fc.array(boolCell, { minLength: n, maxLength: n }),
  }),
);

/** A small integer multiplier (stays well within i32 range given col values ≤30). */
const smallMul = fc.integer({ min: 1, max: 4 });
/** A small integer addend. */
const smallAdd = fc.integer({ min: -10, max: 10 });

// ── Pipeline step types ───────────────────────────────────────────────────────

type StepKind =
  | { kind: 'filterAGt'; val: number }
  | { kind: 'filterALt'; val: number }
  | { kind: 'filterBGt'; val: number }
  | { kind: 'filterBLt'; val: number }
  | { kind: 'filterGEq'; val: string }
  | { kind: 'mulA'; val: number }
  | { kind: 'addA'; val: number }
  | { kind: 'mulB'; val: number }
  | { kind: 'addB'; val: number }
  | { kind: 'sortAsc'; col: 'b' | 'g' }
  | { kind: 'sortDesc'; col: 'b' | 'g' }
  | { kind: 'head'; n: number }
  | { kind: 'slice'; i: number; j: number };

const stepArb: fc.Arbitrary<StepKind> = fc.oneof(
  fc.record({ kind: fc.constant('filterAGt' as const), val: fc.integer({ min: -50, max: 50 }) }),
  fc.record({ kind: fc.constant('filterALt' as const), val: fc.integer({ min: -50, max: 50 }) }),
  fc.record({ kind: fc.constant('filterBGt' as const), val: fc.integer({ min: -25, max: 25 }) }),
  fc.record({ kind: fc.constant('filterBLt' as const), val: fc.integer({ min: -25, max: 25 }) }),
  fc.record({ kind: fc.constant('filterGEq' as const), val: fc.constantFrom(...STR_VALS) }),
  fc.record({ kind: fc.constant('mulA' as const), val: smallMul }),
  fc.record({ kind: fc.constant('addA' as const), val: fc.double({ min: -10, max: 10, noNaN: true, noDefaultInfinity: true }) }),
  fc.record({ kind: fc.constant('mulB' as const), val: smallMul }),
  fc.record({ kind: fc.constant('addB' as const), val: smallAdd }),
  fc.record({ kind: fc.constant('sortAsc' as const), col: fc.constantFrom('b' as const, 'g' as const) }),
  fc.record({ kind: fc.constant('sortDesc' as const), col: fc.constantFrom('b' as const, 'g' as const) }),
  fc.record({ kind: fc.constant('head' as const), n: fc.nat({ max: 30 }) }),
  fc.record({
    kind: fc.constant('slice' as const),
    i: fc.nat({ max: 25 }),
    j: fc.nat({ max: 50 }),
  }),
);

const pipelineArb = fc.array(stepArb, { minLength: 0, maxLength: 6 });

// ── Apply a step ──────────────────────────────────────────────────────────────

function applyLibStep(df: DataFrame, step: StepKind): DataFrame {
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
  }
}

function applyOracleStep(frame: OFrame, step: StepKind): OFrame {
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
  }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

const FIXED_COLS = ['a', 'b', 'g', 'flag'] as const;

describe('public API fuzz — random pipelines vs naive oracle', () => {
  it('schema-preserving pipelines (filter/withColumn/sort/head/slice)', () => {
    fc.assert(
      fc.property(frameArb, pipelineArb, ({ a, b, g, flag }, steps) => {
        const init = makeDF(rt, { a, b, g, flag }, { a: 'f64', b: 'i32', g: 'utf8', flag: 'bool' });
        const ownedFrames: DataFrame[] = [init];
        try {
          let current: DataFrame = init;
          let oracle: OFrame = makeOracleFrame(
            a as (number | null)[],
            b as (number | null)[],
            g as (string | null)[],
            flag as (boolean | null)[],
          );

          for (const step of steps) {
            const next = applyLibStep(current, step);
            ownedFrames.push(next);
            oracle = applyOracleStep(oracle, step);
            current = next;
          }

          const libCols = current.toColumns();
          const err = compareFrames(libCols, oracle, [...FIXED_COLS]);
          if (err) throw new Error(`Pipeline mismatch: ${err}\nSteps: ${JSON.stringify(steps)}`);
        } finally {
          for (const df of ownedFrames) df.dispose();
        }
      }),
      { numRuns: NUM_RUNS },
    );
  });

  // ── Shared groupby-agg property (reused by main test + pinned seeds) ──────────
  type AggFn = 'sum' | 'mean' | 'count' | 'min' | 'max';

  function groupbyAggProperty(aggFnArb: fc.Arbitrary<AggFn>) {
    return fc.property(frameArb, pipelineArb, aggFnArb, ({ a, b, g, flag }, steps, aggFn) => {
      const init = makeDF(rt, { a, b, g, flag }, { a: 'f64', b: 'i32', g: 'utf8', flag: 'bool' });
      const ownedFrames: DataFrame[] = [init];
      try {
        let current: DataFrame = init;
        let oracle: OFrame = makeOracleFrame(
          a as (number | null)[],
          b as (number | null)[],
          g as (string | null)[],
          flag as (boolean | null)[],
        );

        for (const step of steps) {
          const next = applyLibStep(current, step);
          ownedFrames.push(next);
          oracle = applyOracleStep(oracle, step);
          current = next;
        }

        // Terminal groupby
        const aggResult = current.groupby('g').agg({ a: aggFn });
        ownedFrames.push(aggResult);
        const oracleAgg = oracleGroupbyAgg(oracle, 'g', 'a', aggFn);

        // Sort lib result by 'g' to match oracle sort
        const sortedAgg = aggResult.sortValues('g', { descending: false });
        ownedFrames.push(sortedAgg);

        const libCols = sortedAgg.toColumns();
        expect(libCols['g']?.length ?? 0, 'group count matches oracle').toBe(oracleAgg.length);

        // Build maps for comparison (key → aggregated value)
        const libMap = new Map<Cell, Cell>();
        const libG = libCols['g'] ?? [];
        const libA = libCols['a'] ?? [];
        for (let i = 0; i < libG.length; i++) {
          libMap.set(libG[i] ?? null, libA[i] ?? null);
        }

        for (const oracleRow of oracleAgg) {
          const oKey = oracleRow['g'] ?? null;
          const oVal = oracleRow['a'] ?? null;
          const lVal = libMap.has(oKey) ? (libMap.get(oKey) ?? null) : undefined;
          if (lVal === undefined) {
            throw new Error(`groupby key ${JSON.stringify(oKey)} missing from library output`);
          }
          // Use relative tolerance for mean/sum (floating point accumulation differs)
          if (aggFn === 'mean' || aggFn === 'sum') {
            if (oVal === null && lVal !== null) {
              throw new Error(`groupby ${aggFn}[${JSON.stringify(oKey)}]: lib=${JSON.stringify(lVal)} oracle=null`);
            }
            if (oVal !== null && lVal === null) {
              throw new Error(`groupby ${aggFn}[${JSON.stringify(oKey)}]: lib=null oracle=${JSON.stringify(oVal)}`);
            }
            if (oVal !== null && lVal !== null) {
              const diff = Math.abs((lVal as number) - (oVal as number));
              const mag = Math.max(Math.abs(oVal as number), 1);
              if (diff / mag > 1e-9) {
                throw new Error(`groupby ${aggFn}[${JSON.stringify(oKey)}]: lib=${lVal} oracle=${oVal} diff=${diff}`);
              }
            }
          } else {
            if (!cellEq(lVal, oVal)) {
              throw new Error(`groupby ${aggFn}[${JSON.stringify(oKey)}]: lib=${JSON.stringify(lVal)} oracle=${JSON.stringify(oVal)}`);
            }
          }
        }
      } finally {
        for (const df of ownedFrames) df.dispose();
      }
    });
  }

  const aggArb: fc.Arbitrary<AggFn> = fc.constantFrom('sum', 'mean', 'count', 'min', 'max');

  it('terminal groupby-agg vs oracle', () => {
    fc.assert(groupbyAggProperty(aggArb), { numRuns: NUM_RUNS });
  });

  // Pinned seeds: these previously exposed the oracle NaN/null conflation bug
  // (oracle excluded NaN from group values, causing count to return 0 for NaN rows).
  // After fixing oracle.ts §4 compliance, both seeds must pass.
  it('pinned seed 141263226 — groupby NaN/null oracle regression', () => {
    fc.assert(groupbyAggProperty(aggArb), { seed: 141263226, numRuns: 50 });
  });

  it('pinned seed 2099073632 — groupby NaN/null oracle regression', () => {
    fc.assert(groupbyAggProperty(aggArb), { seed: 2099073632, numRuns: 50 });
  });

  it('select keeps only the requested columns', () => {
    type ColSubset = ('a' | 'b' | 'g' | 'flag')[];
    const subsetArb: fc.Arbitrary<ColSubset> = fc
      .subarray(['a', 'b', 'g', 'flag'] as const, { minLength: 1 })
      .map((s) => [...s] as ColSubset);

    fc.assert(
      fc.property(frameArb, subsetArb, ({ a, b, g, flag }, subset) => {
        const init = makeDF(rt, { a, b, g, flag }, { a: 'f64', b: 'i32', g: 'utf8', flag: 'bool' });
        try {
          const selected = init.select(subset);
          try {
            const oracle = oracleSelect(
              makeOracleFrame(
                a as (number | null)[],
                b as (number | null)[],
                g as (string | null)[],
                flag as (boolean | null)[],
              ),
              subset,
            );
            expect(selected.columns).toEqual(subset);
            expect(selected.length).toBe(a.length);
            const err = compareFrames(selected.toColumns(), oracle, subset);
            if (err) throw new Error(`select mismatch: ${err}`);
          } finally {
            selected.dispose();
          }
        } finally {
          init.dispose();
        }
      }),
      { numRuns: NUM_RUNS },
    );
  });
});
