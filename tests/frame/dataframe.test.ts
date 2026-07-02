/**
 * DataFrame core API integration tests (spec §4): construction/export, select/drop/
 * withColumn/filter, sortValues, head/tail/slice, shape/columns/dtypes/col, describe,
 * filterFn/mapFn, errors, ownership/dispose, and the table printer.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { loadRuntimeForTest, makeDF } from './helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import { DataFrame, scope } from '../../src/frame/dataframe.js';
import { Series } from '../../src/frame/series.js';
import { FrameError } from '../../src/frame/errors.js';
import { col } from '../../src/expr/index.js';

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

describe('construction & export', () => {
  it('fromColumns round-trips typed arrays and plain arrays with nulls', () => {
    const df = makeDF(rt, {
      a: new Float64Array([1, 2, 3]),
      b: [10, null, 30],
      s: ['x', 'y', null],
    });
    expect(df.shape).toEqual([3, 3]);
    expect(df.columns).toEqual(['a', 'b', 's']);
    expect(df.dtypes).toEqual({ a: 'f64', b: 'f64', s: 'utf8' });
    expect(df.toColumns()).toEqual({ a: [1, 2, 3], b: [10, null, 30], s: ['x', 'y', null] });
    df.dispose();
  });

  it('fromColumns infers i32 override and bool', () => {
    const df = makeDF(rt, { c: [1, 2, 3], flag: [true, false, null] }, { c: 'i32' });
    expect(df.dtypes).toEqual({ c: 'i32', flag: 'bool' });
    expect(df.toColumns()).toEqual({ c: [1, 2, 3], flag: [true, false, null] });
    df.dispose();
  });

  it('fromRecords unions keys and fills missing with null', () => {
    const df = DataFrame.fromRecords(
      [
        { a: 1, b: 'x' },
        { a: 2 },
        { a: 3, b: 'z', c: true },
      ],
      { runtime: rt },
    );
    expect(df.columns).toEqual(['a', 'b', 'c']);
    expect(df.toRecords()).toEqual([
      { a: 1, b: 'x', c: null },
      { a: 2, b: null, c: null },
      { a: 3, b: 'z', c: true },
    ]);
    df.dispose();
  });

  it('rejects ragged columns', () => {
    expect(() => makeDF(rt, { a: [1, 2], b: [1] })).toThrow(FrameError);
  });
});

describe('projection & assignment', () => {
  it('select / drop reorder and remove columns (zero-copy share)', () => {
    const df = makeDF(rt, { a: [1, 2], b: [3, 4], c: [5, 6] });
    const sel = df.select(['c', 'a']);
    expect(sel.columns).toEqual(['c', 'a']);
    expect(sel.toColumns()).toEqual({ c: [5, 6], a: [1, 2] });
    const dropped = df.drop(['b']);
    expect(dropped.columns).toEqual(['a', 'c']);
    sel.dispose();
    dropped.dispose();
    // parent still valid after children disposed (refcount)
    expect(df.toColumns().a).toEqual([1, 2]);
    df.dispose();
  });

  it('withColumn computes an expression and shares untouched columns', () => {
    const df = makeDF(rt, { a: [1, 2, 3], b: [10, 20, 30] }, { a: 'f64', b: 'f64' });
    const df2 = df.withColumn('c', col('a').add(col('b')));
    expect(df2.columns).toEqual(['a', 'b', 'c']);
    expect(df2.toColumns().c).toEqual([11, 22, 33]);
    // assign alias + raw-array assignment
    const df3 = df.assign('d', [100, 200, 300], { dtype: 'i32' });
    expect(df3.toColumns().d).toEqual([100, 200, 300]);
    df.dispose(); df2.dispose(); df3.dispose();
  });

  it('withColumn replaces an existing column in place (order preserved)', () => {
    const df = makeDF(rt, { a: [1, 2], b: [3, 4] }, { a: 'f64', b: 'f64' });
    const df2 = df.withColumn('a', col('a').mul(10));
    expect(df2.columns).toEqual(['a', 'b']);
    expect(df2.toColumns().a).toEqual([10, 20]);
    df.dispose(); df2.dispose();
  });

  it('withColumn broadcasts a scalar aggregation', () => {
    const df = makeDF(rt, { a: [1, 2, 3, 4] }, { a: 'f64' });
    const df2 = df.withColumn('m', col('a').sum());
    expect(df2.toColumns().m).toEqual([10, 10, 10, 10]);
    df.dispose(); df2.dispose();
  });
});

describe('filter', () => {
  it('filters by expression (null/false dropped)', () => {
    const df = makeDF(rt, { a: [1, 2, 3, 4, 5], b: ['a', 'b', 'c', 'd', 'e'] }, { a: 'i32', b: 'utf8' });
    const out = df.filter(col('a').gt(2).and(col('a').lt(5)));
    expect(out.toColumns()).toEqual({ a: [3, 4], b: ['c', 'd'] });
    out.dispose(); df.dispose();
  });

  it('filter drops null-predicate rows', () => {
    const df = makeDF(rt, { a: [1, null, 3] }, { a: 'i32' });
    const out = df.filter(col('a').gt(0));
    expect(out.toColumns().a).toEqual([1, 3]);
    out.dispose(); df.dispose();
  });

  it('filterFn / mapFn escape hatch', () => {
    const df = makeDF(rt, { a: [1, 2, 3, 4], s: ['w', 'x', 'y', 'z'] }, { a: 'i32', s: 'utf8' });
    const out = df.filterFn((r) => (r.a as number) % 2 === 0);
    expect(out.toColumns()).toEqual({ a: [2, 4], s: ['x', 'z'] });
    const mapped = df.mapFn((r) => `${r.s}${r.a}`);
    expect(mapped).toEqual(['w1', 'x2', 'y3', 'z4']);
    out.dispose(); df.dispose();
  });
});

describe('sortValues', () => {
  it('single numeric key, ascending, nulls last, stable', () => {
    const df = makeDF(rt, { a: [3, null, 1, 2, 1], id: [0, 1, 2, 3, 4] }, { a: 'i32', id: 'i32' });
    const out = df.sortValues('a');
    expect(out.toColumns().a).toEqual([1, 1, 2, 3, null]);
    // stable: the two 1s keep original order (ids 2 then 4)
    expect(out.toColumns().id).toEqual([2, 4, 3, 0, 1]);
    out.dispose(); df.dispose();
  });

  it('descending keeps nulls last', () => {
    const df = makeDF(rt, { a: [3, null, 1, 2] }, { a: 'f64' });
    const out = df.sortValues('a', { descending: true });
    expect(out.toColumns().a).toEqual([3, 2, 1, null]);
    out.dispose(); df.dispose();
  });

  it('utf8 sorts lexicographically', () => {
    const df = makeDF(rt, { s: ['banana', 'apple', 'cherry', 'apple'] }, { s: 'utf8' });
    const out = df.sortValues('s');
    expect(out.toColumns().s).toEqual(['apple', 'apple', 'banana', 'cherry']);
    out.dispose(); df.dispose();
  });

  it('multi-key sort (last-key-first threading)', () => {
    const df = makeDF(
      rt,
      { g: ['b', 'a', 'b', 'a'], v: [2, 2, 1, 1] },
      { g: 'utf8', v: 'i32' },
    );
    const out = df.sortValues(['g', 'v']);
    expect(out.toColumns()).toEqual({ g: ['a', 'a', 'b', 'b'], v: [1, 2, 1, 2] });
    out.dispose(); df.dispose();
  });
});

describe('slicing (zero-copy)', () => {
  it('head / tail / slice', () => {
    const df = makeDF(rt, { a: [0, 1, 2, 3, 4, 5] }, { a: 'i32' });
    expect(df.head(2).toColumns().a).toEqual([0, 1]);
    expect(df.tail(2).toColumns().a).toEqual([4, 5]);
    expect(df.slice(2, 4).toColumns().a).toEqual([2, 3]);
    expect(df.slice(4, 100).toColumns().a).toEqual([4, 5]);
    df.dispose();
  });
});

describe('col / Series / describe', () => {
  it('col returns a zero-copy Series', () => {
    const df = makeDF(rt, { a: [1, 2, 3] }, { a: 'f64' });
    const s = df.col('a');
    expect(s).toBeInstanceOf(Series);
    expect(s.name).toBe('a');
    expect(s.dtype).toBe('f64');
    expect(s.toArray()).toEqual([1, 2, 3]);
    expect(Array.from(s.values())).toEqual([1, 2, 3]);
    df.dispose();
  });

  it('describe summarises numeric columns', () => {
    const df = makeDF(rt, { a: [1, 2, 3, 4], s: ['x', 'y', 'z', 'w'] }, { a: 'f64', s: 'utf8' });
    const d = df.describe();
    expect(d.columns).toEqual(['statistic', 'a']);
    const cols = d.toColumns();
    expect(cols.statistic).toEqual(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']);
    const a = cols.a as number[];
    expect(a[0]).toBe(4); // count
    expect(a[1]).toBeCloseTo(2.5); // mean
    expect(a[3]).toBe(1); // min
    expect(a[7]).toBe(4); // max
    d.dispose(); df.dispose();
  });
});

describe('errors', () => {
  it('unknown column suggests the nearest name', () => {
    const df = makeDF(rt, { alpha: [1], beta: [2] }, { alpha: 'i32', beta: 'i32' });
    expect(() => df.select(['alpah'])).toThrow(/Did you mean 'alpha'/);
    df.dispose();
  });

  it('sortValues on a missing column throws FrameError', () => {
    const df = makeDF(rt, { a: [1] }, { a: 'i32' });
    expect(() => df.sortValues('z')).toThrow(FrameError);
    df.dispose();
  });
});

describe('printer', () => {
  it('renders an aligned preview with dtype header and footer', () => {
    const df = makeDF(rt, { a: [1, 2, 3], name: ['al', 'bo', null] }, { a: 'i32', name: 'utf8' });
    const s = df.toString();
    expect(s).toContain('a');
    expect(s).toContain('i32');
    expect(s).toContain('utf8');
    expect(s).toContain('null');
    expect(s).toContain('[3 rows × 2 columns]');
    df.dispose();
  });

  it('truncates large frames with an ellipsis row', () => {
    const a = Array.from({ length: 100 }, (_, i) => i);
    const df = makeDF(rt, { a }, { a: 'i32' });
    const s = df.toString();
    expect(s).toContain('…');
    expect(s).toContain('[100 rows × 1 columns]');
    df.dispose();
  });
});

describe('scope helper', () => {
  it('disposes tracked frames', () => {
    const base = makeDF(rt, { a: [1, 2, 3] }, { a: 'f64' });
    const total = scope((track) => {
      const filtered = track(base.filter(col('a').gt(1)));
      return filtered.toColumns().a;
    });
    expect(total).toEqual([2, 3]);
    base.dispose();
  });
});
