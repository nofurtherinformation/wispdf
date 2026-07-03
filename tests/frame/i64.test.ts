/**
 * i64 first-class support through the frame layer (v2.4).
 * Covers: construction, toColumns, expr/arithmetic, cmp, cast, agg, sort, groupby,
 * filterFn/mapFn row proxy, describe, printer.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { loadRuntimeForTest, makeDF } from './helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import { DataFrame } from '../../src/frame/dataframe.js';
import { col, lit } from '../../src/expr/index.js';

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

// ---- construction & export ----

describe('i64 construction & export', () => {
  it('infers i64 from BigInt64Array fast path', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([1n, 2n, 3n]) });
    expect(df.dtypes).toEqual({ x: 'i64' });
    expect(df.toColumns()).toEqual({ x: [1n, 2n, 3n] });
    df.dispose();
  });

  it('infers i64 from bigint values in plain array', () => {
    const df = makeDF(rt, { v: [10n, null, 30n] });
    expect(df.dtypes).toEqual({ v: 'i64' });
    expect(df.toColumns()).toEqual({ v: [10n, null, 30n] });
    df.dispose();
  });

  it('fromRecords infers i64 from bigint fields', () => {
    const df = DataFrame.fromRecords(
      [{ id: 100n }, { id: 200n }, { id: null }],
      { runtime: rt },
    );
    expect(df.dtypes).toEqual({ id: 'i64' });
    expect(df.toColumns()).toEqual({ id: [100n, 200n, null] });
    df.dispose();
  });

  it('INT64 boundary values round-trip', () => {
    const INT64_MAX = 9223372036854775807n;
    const INT64_MIN = -9223372036854775808n;
    const df = makeDF(rt, { b: BigInt64Array.from([INT64_MAX, INT64_MIN, 0n]) });
    expect(df.toColumns()).toEqual({ b: [INT64_MAX, INT64_MIN, 0n] });
    df.dispose();
  });
});

// ---- arithmetic & comparison exprs ----

describe('i64 arithmetic exprs', () => {
  it('add i64 col + i64 col', () => {
    const df = makeDF(rt, { a: BigInt64Array.from([1n, 2n, 3n]), b: BigInt64Array.from([10n, 20n, 30n]) });
    const r = df.withColumn('c', col('a').add(col('b'))).toColumns();
    expect(r['c']).toEqual([11n, 22n, 33n]);
    df.dispose();
  });

  it('mul i64 col * literal bigint', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([1n, 2n, 3n]) });
    const r = df.withColumn('y', col('x').mul(lit(10n))).toColumns();
    expect(r['y']).toEqual([10n, 20n, 30n]);
    df.dispose();
  });

  it('neg i64', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([1n, -2n, 0n]) });
    const r = df.withColumn('y', col('x').neg()).toColumns();
    expect(r['y']).toEqual([-1n, 2n, 0n]);
    df.dispose();
  });

  it('i64 + i32 widens to i64', () => {
    const df = makeDF(rt, { a: BigInt64Array.from([100n, 200n]), b: new Int32Array([1, 2]) });
    const r = df.withColumn('c', col('a').add(col('b'))).toColumns();
    expect(r['c']).toEqual([101n, 202n]);
    df.dispose();
  });

  it('i64 + f64 widens to f64', () => {
    const df = makeDF(rt, { a: BigInt64Array.from([1n, 2n]), b: new Float64Array([0.5, 1.5]) });
    const r = df.withColumn('c', col('a').add(col('b'))).toColumns();
    expect(r['c']).toEqual([1.5, 3.5]);
    df.dispose();
  });

  it('i64 gt comparison returns bool column', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([1n, 5n, 3n]) });
    const r = df.withColumn('ok', col('x').gt(lit(2n))).toColumns();
    expect(r['ok']).toEqual([false, true, true]);
    df.dispose();
  });
});

// ---- aggregations ----
// Aggregations are evaluated via withColumn (scalar broadcast) — no direct .sum() on Series.
function agg(df: ReturnType<typeof makeDF>, name: string, expr: ReturnType<typeof col>): unknown {
  const r = df.withColumn('__agg', expr);
  const v = r.toColumns()['__agg']![0];
  r.dispose();
  return v;
}

describe('i64 aggregations', () => {
  it('sum(i64) returns bigint', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([1n, 2n, 3n, 4n, 5n]) });
    const s = agg(df, 'x', col('x').sum());
    expect(s).toBe(15n);
    expect(typeof s).toBe('bigint');
    df.dispose();
  });

  it('sum(i64) with nulls skips nulls', () => {
    const df = makeDF(rt, { x: [1n, null, 3n] });
    expect(agg(df, 'x', col('x').sum())).toBe(4n);
    df.dispose();
  });

  it('mean(i64) returns number (f64)', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([10n, 20n, 30n]) });
    const m = agg(df, 'x', col('x').mean());
    expect(m).toBe(20);
    expect(typeof m).toBe('number');
    df.dispose();
  });

  it('min/max(i64) return bigint', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([5n, 1n, 9n, 3n]) });
    expect(agg(df, 'x', col('x').min())).toBe(1n);
    expect(agg(df, 'x', col('x').max())).toBe(9n);
    df.dispose();
  });

  it('min(i64) with all nulls returns null', () => {
    const df = makeDF(rt, { x: [null, null] as (bigint | null)[] });
    expect(agg(df, 'x', col('x').min())).toBeNull();
    df.dispose();
  });
});

// ---- cast ----

describe('i64 cast', () => {
  it('cast i64 -> f64', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([1n, 2n, 3n]) });
    const r = df.withColumn('y', col('x').cast('f64')).toColumns();
    expect(r['y']).toEqual([1, 2, 3]);
    df.dispose();
  });

  it('cast f64 -> i64 (truncates)', () => {
    const df = makeDF(rt, { x: new Float64Array([2.9, -2.9, 1.0]) });
    const r = df.withColumn('y', col('x').cast('i64')).toColumns();
    expect(r['y']).toEqual([2n, -2n, 1n]);
    df.dispose();
  });

  it('cast f64 -> i64 (range null for out-of-range)', () => {
    const df = makeDF(rt, { x: new Float64Array([9.9e18]) });
    const r = df.withColumn('y', col('x').cast('i64')).toColumns();
    expect(r['y']).toEqual([null]);
    df.dispose();
  });

  it('cast i32 -> i64 (sign-extend)', () => {
    const df = makeDF(rt, { x: new Int32Array([-1, 2147483647]) });
    const r = df.withColumn('y', col('x').cast('i64')).toColumns();
    expect(r['y']).toEqual([-1n, 2147483647n]);
    df.dispose();
  });

  it('cast i64 -> i32 (wrap-truncate)', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([4294967297n]) }); // 2^32+1 -> 1
    const r = df.withColumn('y', col('x').cast('i32')).toColumns();
    expect(r['y']).toEqual([1]);
    df.dispose();
  });
});

// ---- sort ----

describe('i64 sortValues', () => {
  it('sorts i64 ascending with nulls last', () => {
    const df = makeDF(rt, { x: [5n, null, 1n, 3n] });
    const sorted = df.sortValues('x').toColumns();
    expect(sorted['x']).toEqual([1n, 3n, 5n, null]);
    df.dispose();
  });

  it('sorts i64 descending', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([3n, 1n, 4n, 1n, 5n]) });
    const sorted = df.sortValues('x', { descending: true }).toColumns();
    expect(sorted['x']).toEqual([5n, 4n, 3n, 1n, 1n]);
    df.dispose();
  });
});

// ---- filter ----

describe('i64 filter', () => {
  it('filter on i64 column', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([1n, 2n, 3n, 4n]) });
    const filtered = df.filter(col('x').gt(lit(2n))).toColumns();
    expect(filtered['x']).toEqual([3n, 4n]);
    df.dispose();
  });
});

// ---- groupby ----

describe('i64 groupby', () => {
  it('groupby sum of i64', () => {
    const df = makeDF(rt, {
      k: ['a', 'b', 'a', 'b'],
      v: BigInt64Array.from([1n, 2n, 3n, 4n]),
    });
    const gb = df.groupby('k').agg({ v: 'sum' }).sortValues('k');
    const r = gb.toColumns();
    expect(r['k']).toEqual(['a', 'b']);
    expect(r['v']).toEqual([4n, 6n]);
    gb.dispose();
    df.dispose();
  });

  it('groupby mean of i64 returns f64', () => {
    const df = makeDF(rt, {
      k: ['a', 'a'],
      v: BigInt64Array.from([10n, 30n]),
    });
    const gb = df.groupby('k').agg({ v: 'mean' });
    const r = gb.toColumns();
    expect(r['v']).toEqual([20]);
    gb.dispose();
    df.dispose();
  });

  it('groupby min/max of i64 returns bigint', () => {
    const df = makeDF(rt, {
      k: ['a', 'a', 'b'],
      v: BigInt64Array.from([5n, 1n, 9n]),
    });
    const min = df.groupby('k').agg({ v: 'min' }).sortValues('k');
    const max = df.groupby('k').agg({ v: 'max' }).sortValues('k');
    expect(min.toColumns()['v']).toEqual([1n, 9n]);
    expect(max.toColumns()['v']).toEqual([5n, 9n]);
    min.dispose();
    max.dispose();
    df.dispose();
  });
});

// ---- filterFn / mapFn row proxy ----

describe('i64 row proxy', () => {
  it('filterFn receives bigint for i64 column', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([1n, 2n, 3n]) });
    const filtered = df.filterFn((row) => (row['x'] as bigint) > 1n);
    expect(filtered.toColumns()).toEqual({ x: [2n, 3n] });
    filtered.dispose();
    df.dispose();
  });

  it('mapFn returns bigint for i64 column', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([1n, 2n, 3n]) });
    const vals: bigint[] = [];
    df.mapFn((row) => { vals.push(row['x'] as bigint); return row; });
    expect(vals).toEqual([1n, 2n, 3n]);
    df.dispose();
  });
});

// ---- printer ----

describe('i64 printer', () => {
  it('prints bigint without n suffix', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([1n, -999n]) });
    const s = df.toString();
    expect(s).toContain('1');
    expect(s).toContain('-999');
    // Cell values must not have trailing 'n' (e.g. "1n" or "-999n" would be wrong)
    expect(s).not.toContain('1n');
    expect(s).not.toContain('-999n');
    df.dispose();
  });
});

// ---- describe ----

describe('i64 describe', () => {
  it('describe includes i64 columns without throwing', () => {
    const df = makeDF(rt, { x: BigInt64Array.from([10n, 20n, 30n]) });
    const d = df.describe();
    expect(d.columns).toContain('x');
    d.dispose();
    df.dispose();
  });
});

// ---- fillNull ----

describe('i64 fillNull', () => {
  it('fillNull(bigint) fills null slots', () => {
    const df = makeDF(rt, { x: [1n, null, 3n] });
    // fillNull takes a scalar value directly (not lit())
    const r = df.withColumn('x', col('x').fillNull(0n)).toColumns();
    expect(r['x']).toEqual([1n, 0n, 3n]);
    df.dispose();
  });
});
