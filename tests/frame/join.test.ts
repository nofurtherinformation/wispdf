/**
 * Join integration tests (spec §4; ADR-005; dtypes.md §4.5): inner/left, numeric and
 * utf8 (dictionary-unified) keys, multi-key, null-key exclusion, and name collisions.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { loadRuntimeForTest, makeDF } from './helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import type { Cell } from '../../src/memory/column.js';

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

function sortRecs(recs: Array<Record<string, Cell>>, key: string): Array<Record<string, Cell>> {
  return [...recs].sort((a, b) => (Number(a[key]) - Number(b[key])));
}

describe('inner join', () => {
  it('numeric keys, keeps only matches', () => {
    const left = makeDF(rt, { id: [1, 2, 3], l: ['a', 'b', 'c'] }, { id: 'i32', l: 'utf8' });
    const right = makeDF(rt, { id: [2, 3, 4], r: [20, 30, 40] }, { id: 'i32', r: 'f64' });
    const out = left.join(right, { on: 'id', how: 'inner' });
    expect(out.columns).toEqual(['id', 'l', 'r']);
    expect(sortRecs(out.toRecords(), 'id')).toEqual([
      { id: 2, l: 'b', r: 20 },
      { id: 3, l: 'c', r: 30 },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('utf8 keys (dictionary unified across frames)', () => {
    const left = makeDF(rt, { k: ['cat', 'dog', 'fish'], n: [1, 2, 3] }, { k: 'utf8', n: 'i32' });
    const right = makeDF(rt, { k: ['dog', 'cat'], m: [200, 100] }, { k: 'utf8', m: 'i32' });
    const out = left.join(right, { on: 'k' });
    const r = out.toRecords().sort((a, b) => (a.k! < b.k! ? -1 : 1));
    expect(r).toEqual([
      { k: 'cat', n: 1, m: 100 },
      { k: 'dog', n: 2, m: 200 },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('duplicate right matches expand rows', () => {
    const left = makeDF(rt, { id: [1, 2] }, { id: 'i32' });
    const right = makeDF(rt, { id: [1, 1, 2], v: [10, 11, 20] }, { id: 'i32', v: 'i32' });
    const out = left.join(right, { on: 'id' });
    expect(sortRecs(out.toRecords(), 'v')).toEqual([
      { id: 1, v: 10 },
      { id: 1, v: 11 },
      { id: 2, v: 20 },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });
});

describe('left join', () => {
  it('unmatched left rows get null right columns', () => {
    const left = makeDF(rt, { id: [1, 2, 3], l: ['a', 'b', 'c'] }, { id: 'i32', l: 'utf8' });
    const right = makeDF(rt, { id: [2], r: [99] }, { id: 'i32', r: 'f64' });
    const out = left.join(right, { on: 'id', how: 'left' });
    expect(sortRecs(out.toRecords(), 'id')).toEqual([
      { id: 1, l: 'a', r: null },
      { id: 2, l: 'b', r: 99 },
      { id: 3, l: 'c', r: null },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('null keys never match', () => {
    const left = makeDF(rt, { id: [1, null], l: ['a', 'b'] }, { id: 'i32', l: 'utf8' });
    const right = makeDF(rt, { id: [1, null], r: [10, 20] }, { id: 'i32', r: 'i32' });
    const out = left.join(right, { on: 'id', how: 'left' });
    expect(sortRecs(out.toRecords().filter((x) => x.id !== null), 'id')).toEqual([
      { id: 1, l: 'a', r: 10 },
    ]);
    // the null-key left row is present but unmatched
    const nullRow = out.toRecords().find((x) => x.id === null)!;
    expect(nullRow.r).toBeNull();
    out.dispose(); left.dispose(); right.dispose();
  });
});

describe('join column naming', () => {
  it('suffixes colliding right columns with _right', () => {
    const left = makeDF(rt, { id: [1, 2], v: [1, 2] }, { id: 'i32', v: 'i32' });
    const right = makeDF(rt, { id: [1, 2], v: [10, 20] }, { id: 'i32', v: 'i32' });
    const out = left.join(right, { on: 'id' });
    expect(out.columns).toEqual(['id', 'v', 'v_right']);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('errors on dtype-mismatched key', () => {
    const left = makeDF(rt, { id: [1] }, { id: 'i32' });
    const right = makeDF(rt, { id: [1] }, { id: 'f64' });
    expect(() => left.join(right, { on: 'id' })).toThrow(/dtype mismatch/);
    out(left, right);
  });
});

function out(...dfs: { dispose(): void }[]): void {
  for (const d of dfs) d.dispose();
}
