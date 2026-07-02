/**
 * GroupBy.agg integration tests (spec §4; dtypes.md §4.3/§4.4/§4.5): every aggregation,
 * string / array / expr agg forms, null-key grouping, and multi-key grouping.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { loadRuntimeForTest, makeDF } from './helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import type { Cell } from '../../src/memory/column.js';
import { col } from '../../src/expr/index.js';

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

/** Sort result rows by the group key for order-independent comparison. */
function byKey(records: Array<Record<string, Cell>>, key: string): Array<Record<string, Cell>> {
  return [...records].sort((a, b) => (a[key]! < b[key]! ? -1 : a[key]! > b[key]! ? 1 : 0));
}

describe('groupby single key', () => {
  it('sum with string agg form, key column first', () => {
    const df = makeDF(
      rt,
      { g: ['a', 'b', 'a', 'b', 'a'], v: [1, 10, 2, 20, 3] },
      { g: 'utf8', v: 'f64' },
    );
    const out = df.groupby('g').agg({ v: 'sum' });
    expect(out.columns).toEqual(['g', 'v']);
    expect(byKey(out.toRecords(), 'g')).toEqual([
      { g: 'a', v: 6 },
      { g: 'b', v: 30 },
    ]);
    out.dispose(); df.dispose();
  });

  it('multiple ops via array form → col_op names', () => {
    const df = makeDF(rt, { g: ['a', 'a', 'b'], v: [1, 3, 5] }, { g: 'utf8', v: 'f64' });
    const out = df.groupby('g').agg({ v: ['mean', 'max', 'count'] });
    expect(out.columns).toEqual(['g', 'v_mean', 'v_max', 'v_count']);
    const r = byKey(out.toRecords(), 'g');
    expect(r[0]).toEqual({ g: 'a', v_mean: 2, v_max: 3, v_count: 2 });
    expect(r[1]).toEqual({ g: 'b', v_mean: 5, v_max: 5, v_count: 1 });
    out.dispose(); df.dispose();
  });

  it('expr agg form names by the output key', () => {
    const df = makeDF(rt, { g: ['a', 'a', 'b'], v: [1, 2, 4] }, { g: 'utf8', v: 'f64' });
    const out = df.groupby('g').agg({ doubled: col('v').mul(2).sum() });
    expect(out.columns).toEqual(['g', 'doubled']);
    const r = byKey(out.toRecords(), 'g');
    expect(r).toEqual([
      { g: 'a', doubled: 6 },
      { g: 'b', doubled: 8 },
    ]);
    out.dispose(); df.dispose();
  });

  it('size counts rows incl nulls; count skips nulls', () => {
    const df = makeDF(rt, { g: ['a', 'a', 'a'], v: [1, null, 3] }, { g: 'utf8', v: 'f64' });
    const out = df.groupby('g').agg({ n: 'size', v: 'count' });
    expect(out.toRecords()).toEqual([{ g: 'a', n: 3, v: 2 }]);
    out.dispose(); df.dispose();
  });

  it('skips nulls in sum/mean/min/max', () => {
    const df = makeDF(rt, { g: ['a', 'a', 'a'], v: [2, null, 4] }, { g: 'utf8', v: 'f64' });
    const out = df.groupby('g').agg({ s: col('v').sum(), m: col('v').mean(), lo: col('v').min(), hi: col('v').max() });
    expect(out.toRecords()).toEqual([{ g: 'a', s: 6, m: 3, lo: 2, hi: 4 }]);
    out.dispose(); df.dispose();
  });

  it('std / var / nunique / first / last', () => {
    const df = makeDF(rt, { g: ['a', 'a', 'a', 'a'], v: [1, 2, 3, 3] }, { g: 'utf8', v: 'f64' });
    const out = df.groupby('g').agg({
      sd: col('v').std(),
      va: col('v').var(),
      nu: col('v').nunique(),
      fi: col('v').first(),
      la: col('v').last(),
    });
    const r = out.toRecords()[0]!;
    expect(r.va).toBeCloseTo(0.9166666, 5);
    expect(r.sd).toBeCloseTo(Math.sqrt(0.9166666), 5);
    expect(r.nu).toBe(3);
    expect(r.fi).toBe(1);
    expect(r.la).toBe(3);
    out.dispose(); df.dispose();
  });

  it('null key forms its own group', () => {
    const df = makeDF(rt, { g: ['a', null, 'a', null], v: [1, 2, 3, 4] }, { g: 'utf8', v: 'f64' });
    const out = df.groupby('g').agg({ v: 'sum' });
    const recs = out.toRecords();
    expect(recs.find((r) => r.g === 'a')!.v).toBe(4);
    expect(recs.find((r) => r.g === null)!.v).toBe(6);
    out.dispose(); df.dispose();
  });
});

describe('groupby multi-key', () => {
  it('groups on two keys', () => {
    const df = makeDF(
      rt,
      { a: ['x', 'x', 'y', 'y'], b: [1, 1, 1, 2], v: [10, 20, 30, 40] },
      { a: 'utf8', b: 'i32', v: 'f64' },
    );
    const out = df.groupby(['a', 'b']).agg({ v: 'sum' });
    expect(out.columns).toEqual(['a', 'b', 'v']);
    const r = out.toRecords().sort((x, y) => `${x.a}${x.b}` < `${y.a}${y.b}` ? -1 : 1);
    expect(r).toEqual([
      { a: 'x', b: 1, v: 30 },
      { a: 'y', b: 1, v: 30 },
      { a: 'y', b: 2, v: 40 },
    ]);
    out.dispose(); df.dispose();
  });
});
