/**
 * Unification-free utf8 join tests (ABI §12, CP.2).
 *
 * THE regression-risk test: two sides' dictionaries have different slot orders
 * for the same values.  The unification-free path must still match correctly —
 * it hashes raw UTF-8 bytes per slot (hash_utf8_dict), so "cat" always hashes
 * to the same value regardless of which slot it occupies in a dictionary.
 *
 * Also covers: null keys never match, left-join -1 nulls, multi-key joins
 * including utf8 keys, and property-based comparison with a JS oracle join.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import fc from 'fast-check';
import { loadRuntimeForTest, makeDF } from './helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import type { Cell } from '../../src/memory/column.js';

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Sort records by a string key then by a numeric key (stable secondary). */
function sortByStrThenNum(
  recs: Array<Record<string, Cell>>,
  strKey: string,
  numKey: string,
): Array<Record<string, Cell>> {
  return [...recs].sort((a, b) => {
    const as = String(a[strKey] ?? '');
    const bs = String(b[strKey] ?? '');
    if (as < bs) return -1;
    if (as > bs) return 1;
    return Number(a[numKey] ?? 0) - Number(b[numKey] ?? 0);
  });
}

function sortByNum(recs: Array<Record<string, Cell>>, key: string): Array<Record<string, Cell>> {
  return [...recs].sort((a, b) => Number(a[key] ?? 0) - Number(b[key] ?? 0));
}

function sortByStr(recs: Array<Record<string, Cell>>, key: string): Array<Record<string, Cell>> {
  return [...recs].sort((a, b) => {
    const av = String(a[key] ?? '');
    const bv = String(b[key] ?? '');
    return av < bv ? -1 : av > bv ? 1 : 0;
  });
}

// ---------------------------------------------------------------------------
// THE regression-risk test: different dictionary slot orderings
// ---------------------------------------------------------------------------

describe('utf8 join — different dictionary slot orders (the regression risk)', () => {
  it('inner join: left dict=[cat,dog,fish], right dict=[fish,cat,dog] → same matches', () => {
    // Left dictionary will be built as ["cat","dog","fish"] (slots 0,1,2)
    // Right dictionary will be built as ["fish","cat","dog"] (slots 0,1,2)
    // Row indices will refer to these different slot positions.
    const left = makeDF(
      rt,
      { k: ['cat', 'dog', 'fish', 'cat'], v: [1, 2, 3, 4] },
      { k: 'utf8', v: 'i32' },
    );
    const right = makeDF(
      rt,
      { k: ['fish', 'cat', 'dog', 'fish'], w: [30, 10, 20, 31] },
      { k: 'utf8', w: 'i32' },
    );
    const out = left.join(right, { on: 'k', how: 'inner' });
    const recs = sortByStrThenNum(out.toRecords(), 'k', 'v');
    // Expected matches:
    //   left:  (cat,1), (dog,2), (fish,3), (cat,4)
    //   right: (fish,30), (cat,10), (dog,20), (fish,31)
    //   cat,1   matches cat,10   → {k:cat, v:1, w:10}
    //   dog,2   matches dog,20   → {k:dog, v:2, w:20}
    //   fish,3  matches fish,30  → {k:fish, v:3, w:30}
    //   fish,3  matches fish,31  → {k:fish, v:3, w:31}
    //   cat,4   matches cat,10   → {k:cat, v:4, w:10}
    // sorted by (k, v): cat/1, cat/4, dog/2, fish/3(30), fish/3(31)
    // (fish rows have same v=3, so their w order matches probe order from join kernel)
    const expected = [
      { k: 'cat',  v: 1, w: 10 },
      { k: 'cat',  v: 4, w: 10 },
      { k: 'dog',  v: 2, w: 20 },
      { k: 'fish', v: 3, w: 30 },
      { k: 'fish', v: 3, w: 31 },
    ];
    expect(recs).toEqual(expected);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('inner join: completely reversed dictionaries still produce correct matches', () => {
    const strs = ['alpha', 'beta', 'gamma', 'delta', 'epsilon'];
    const strsRev = [...strs].reverse(); // ['epsilon','delta','gamma','beta','alpha']
    const left  = makeDF(rt, { k: strs,    v: [1, 2, 3, 4, 5] }, { k: 'utf8', v: 'i32' });
    const right = makeDF(rt, { k: strsRev, w: [50, 40, 30, 20, 10] }, { k: 'utf8', w: 'i32' });
    const out = left.join(right, { on: 'k', how: 'inner' });
    const recs = sortByStr(out.toRecords(), 'k');
    expect(recs).toEqual([
      { k: 'alpha',   v: 1, w: 10 },
      { k: 'beta',    v: 2, w: 20 },
      { k: 'delta',   v: 4, w: 40 },
      { k: 'epsilon', v: 5, w: 50 },
      { k: 'gamma',   v: 3, w: 30 },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('left join: unmatched left rows appear with null right, regardless of dict order', () => {
    // left:  [dog, cat, bird] (slots 0,1,2)
    // right: [cat, elephant]  (slots 0,1) — dog and bird are unmatched
    const left  = makeDF(rt, { k: ['dog', 'cat', 'bird'], v: [1, 2, 3] }, { k: 'utf8', v: 'i32' });
    const right = makeDF(rt, { k: ['cat', 'elephant'],    w: [10, 20]  }, { k: 'utf8', w: 'i32' });
    const out = left.join(right, { on: 'k', how: 'left' });
    const recs = sortByStr(out.toRecords(), 'k');
    expect(recs).toEqual([
      { k: 'bird', v: 3, w: null },
      { k: 'cat',  v: 2, w: 10  },
      { k: 'dog',  v: 1, w: null },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });
});

// ---------------------------------------------------------------------------
// Null key semantics (dtypes.md §4.5: null keys never match)
// ---------------------------------------------------------------------------

describe('utf8 join — null key semantics', () => {
  it('inner join: null keys never match (excluded from output)', () => {
    const left  = makeDF(rt, { k: ['a', null, 'b'],  v: [1, 2, 3] }, { k: 'utf8', v: 'i32' });
    const right = makeDF(rt, { k: ['a', null, 'b'],  w: [10, 20, 30] }, { k: 'utf8', w: 'i32' });
    const out = left.join(right, { on: 'k', how: 'inner' });
    const recs = sortByStr(out.toRecords(), 'k');
    // Only non-null key matches: a→10, b→30
    expect(recs).toEqual([
      { k: 'a', v: 1, w: 10 },
      { k: 'b', v: 3, w: 30 },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('left join: null left keys produce (l_idx, -1) → null right columns', () => {
    const left  = makeDF(rt, { k: ['a', null, 'b'], v: [1, 2, 3] }, { k: 'utf8', v: 'i32' });
    const right = makeDF(rt, { k: ['a', 'b'],       w: [10, 30]  }, { k: 'utf8', w: 'i32' });
    const out = left.join(right, { on: 'k', how: 'left' });
    const recs = sortByStr(out.toRecords().map(r => ({ ...r, k: r.k ?? '\x00' })), 'k')
      .map(r => ({ ...r, k: r.k === '\x00' ? null : r.k }));
    expect(recs).toEqual([
      { k: null, v: 2, w: null },
      { k: 'a',  v: 1, w: 10  },
      { k: 'b',  v: 3, w: 30  },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('all-null key column: inner join produces no rows', () => {
    const left  = makeDF(rt, { k: [null, null], v: [1, 2] }, { k: 'utf8', v: 'i32' });
    const right = makeDF(rt, { k: [null, null], w: [10, 20] }, { k: 'utf8', w: 'i32' });
    const out = left.join(right, { on: 'k', how: 'inner' });
    expect(out.shape[0]).toBe(0);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('all-null key column: left join emits all left rows with null right', () => {
    const left  = makeDF(rt, { k: [null, null], v: [1, 2] }, { k: 'utf8', v: 'i32' });
    const right = makeDF(rt, { k: ['x'],        w: [99]   }, { k: 'utf8', w: 'i32' });
    const out = left.join(right, { on: 'k', how: 'left' });
    expect(out.shape[0]).toBe(2);
    const recs = out.toRecords();
    expect(recs.every(r => r['w'] === null)).toBe(true);
    out.dispose(); left.dispose(); right.dispose();
  });
});

// ---------------------------------------------------------------------------
// Multi-key joins including utf8 keys
// ---------------------------------------------------------------------------

describe('utf8 join — multi-key (utf8 + numeric)', () => {
  it('multi-key inner join (utf8 + i32): only rows matching both keys appear', () => {
    const left  = makeDF(
      rt,
      { name: ['alice', 'alice', 'bob', 'carol'], age: [30, 25, 30, 25], v: [1, 2, 3, 4] },
      { name: 'utf8', age: 'i32', v: 'i32' },
    );
    const right = makeDF(
      rt,
      { name: ['alice', 'bob', 'carol', 'alice'], age: [30, 30, 25, 25], w: [10, 30, 40, 20] },
      { name: 'utf8', age: 'i32', w: 'i32' },
    );
    const out = left.join(right, { on: ['name', 'age'], how: 'inner' });
    const recs = sortByNum(out.toRecords(), 'v');
    // alice+30 matches alice+30 w=10
    // alice+25 matches alice+25 w=20
    // bob+30   matches bob+30   w=30
    // carol+25 matches carol+25 w=40
    expect(recs).toEqual([
      { name: 'alice', age: 30, v: 1, w: 10 },
      { name: 'alice', age: 25, v: 2, w: 20 },
      { name: 'bob',   age: 30, v: 3, w: 30 },
      { name: 'carol', age: 25, v: 4, w: 40 },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('multi-key join: rows that match on name but not age are excluded (inner)', () => {
    const left  = makeDF(rt, { name: ['alice'], age: [30], v: [1] }, { name: 'utf8', age: 'i32', v: 'i32' });
    const right = makeDF(rt, { name: ['alice'], age: [99], w: [99] }, { name: 'utf8', age: 'i32', w: 'i32' });
    const out = left.join(right, { on: ['name', 'age'], how: 'inner' });
    expect(out.shape[0]).toBe(0);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('multi-key with reversed utf8 dict order still matches correctly', () => {
    // Keys are (name, id) — names have reversed slot order between sides
    const leftNames  = ['alice', 'bob', 'carol'];
    const rightNames = ['carol', 'alice', 'bob'];
    const left  = makeDF(
      rt,
      { name: leftNames,  id: [1, 2, 3], v: [10, 20, 30] },
      { name: 'utf8', id: 'i32', v: 'i32' },
    );
    const right = makeDF(
      rt,
      { name: rightNames, id: [3, 1, 2], w: [300, 100, 200] },
      { name: 'utf8', id: 'i32', w: 'i32' },
    );
    const out = left.join(right, { on: ['name', 'id'], how: 'inner' });
    const recs = sortByStr(out.toRecords(), 'name');
    expect(recs).toEqual([
      { name: 'alice', id: 1, v: 10, w: 100 },
      { name: 'bob',   id: 2, v: 20, w: 200 },
      { name: 'carol', id: 3, v: 30, w: 300 },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe('utf8 join — edge cases', () => {
  it('empty left side → empty output (inner and left)', () => {
    const left  = makeDF(rt, { k: [] as string[], v: [] as number[] }, { k: 'utf8', v: 'i32' });
    const right = makeDF(rt, { k: ['a'],          w: [1]             }, { k: 'utf8', w: 'i32' });
    const inner = left.join(right, { on: 'k', how: 'inner' });
    const leftJ = left.join(right, { on: 'k', how: 'left'  });
    expect(inner.shape[0]).toBe(0);
    expect(leftJ.shape[0]).toBe(0);
    inner.dispose(); leftJ.dispose(); left.dispose(); right.dispose();
  });

  it('empty right side → inner empty, left emits all left rows with null right', () => {
    const left  = makeDF(rt, { k: ['a', 'b'], v: [1, 2] }, { k: 'utf8', v: 'i32' });
    const right = makeDF(rt, { k: [] as string[], w: [] as number[] }, { k: 'utf8', w: 'i32' });
    const inner = left.join(right, { on: 'k', how: 'inner' });
    const leftJ = left.join(right, { on: 'k', how: 'left'  });
    expect(inner.shape[0]).toBe(0);
    expect(leftJ.shape[0]).toBe(2);
    expect(leftJ.toRecords().every(r => r['w'] === null)).toBe(true);
    inner.dispose(); leftJ.dispose(); left.dispose(); right.dispose();
  });

  it('both sides empty → inner and left produce empty frames', () => {
    const left  = makeDF(rt, { k: [] as string[] }, { k: 'utf8' });
    const right = makeDF(rt, { k: [] as string[] }, { k: 'utf8' });
    const inner = left.join(right, { on: 'k', how: 'inner' });
    const leftJ = left.join(right, { on: 'k', how: 'left'  });
    expect(inner.shape[0]).toBe(0);
    expect(leftJ.shape[0]).toBe(0);
    inner.dispose(); leftJ.dispose(); left.dispose(); right.dispose();
  });

  it('unicode keys with different dict orders: matches correctly', () => {
    const left  = makeDF(rt, { k: ['日本', '中国', 'USA'], v: [1, 2, 3] }, { k: 'utf8', v: 'i32' });
    const right = makeDF(rt, { k: ['USA', '日本', '中国'], w: [30, 10, 20] }, { k: 'utf8', w: 'i32' });
    const out = left.join(right, { on: 'k', how: 'inner' });
    const recs = sortByStr(out.toRecords(), 'k');
    expect(recs).toEqual([
      { k: 'USA', v: 3, w: 30 },
      { k: '中国', v: 2, w: 20 },
      { k: '日本', v: 1, w: 10 },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });

  it('duplicate right matches expand rows correctly (unification-free path)', () => {
    const left  = makeDF(rt, { k: ['x', 'y'] }, { k: 'utf8' });
    const right = makeDF(rt, { k: ['x', 'x', 'y'], w: [1, 2, 3] }, { k: 'utf8', w: 'i32' });
    const out = left.join(right, { on: 'k', how: 'inner' });
    const recs = sortByStr(sortByNum(out.toRecords(), 'w'), 'k');
    expect(recs).toEqual([
      { k: 'x', w: 1 },
      { k: 'x', w: 2 },
      { k: 'y', w: 3 },
    ]);
    out.dispose(); left.dispose(); right.dispose();
  });
});

// ---------------------------------------------------------------------------
// Property test: unification-free join matches a JS oracle join
// ---------------------------------------------------------------------------

/** JS oracle: inner join two arrays of {k, v} records. */
function jsInnerJoin(
  leftRecs: Array<{ k: string | null; v: number }>,
  rightRecs: Array<{ k: string | null; w: number }>,
): Array<{ k: string | null; v: number; w: number }> {
  const out: Array<{ k: string | null; v: number; w: number }> = [];
  const rightByKey = new Map<string, { k: string | null; w: number }[]>();
  for (const r of rightRecs) {
    if (r.k === null) continue;
    const arr = rightByKey.get(r.k) ?? [];
    arr.push(r);
    rightByKey.set(r.k, arr);
  }
  for (const l of leftRecs) {
    if (l.k === null) continue;
    for (const r of rightByKey.get(l.k) ?? []) {
      out.push({ k: l.k, v: l.v, w: r.w });
    }
  }
  return out;
}

/** Sort by (k, v, w) for stable comparison. */
function sortKVW(
  recs: Array<{ k: string | null; v: number; w: number }>,
): Array<{ k: string | null; v: number; w: number }> {
  return [...recs].sort((a, b) => {
    const ks = String(a.k ?? '').localeCompare(String(b.k ?? ''));
    if (ks !== 0) return ks;
    if (a.v !== b.v) return a.v - b.v;
    return a.w - b.w;
  });
}

describe('utf8 join — property test vs JS oracle', () => {
  it('random string tables with possible nulls: inner join matches oracle', () => {
    // Use a small pool of strings so we get meaningful matches.
    const pool = ['alpha', 'beta', 'gamma', 'delta', null];

    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(...pool), { minLength: 0, maxLength: 20 }),
        fc.array(fc.constantFrom(...pool), { minLength: 0, maxLength: 20 }),
        (leftKeys, rightKeys) => {
          const leftRecs  = leftKeys.map((k, i) => ({ k, v: i }));
          const rightRecs = rightKeys.map((k, i) => ({ k, w: i * 10 }));

          const left  = makeDF(rt, { k: leftKeys,  v: leftRecs.map(r => r.v) }, { k: 'utf8', v: 'i32' });
          const right = makeDF(rt, { k: rightKeys, w: rightRecs.map(r => r.w) }, { k: 'utf8', w: 'i32' });
          try {
            const out = left.join(right, { on: 'k', how: 'inner' });
            try {
              const actual   = sortKVW(out.toRecords() as Array<{ k: string | null; v: number; w: number }>);
              const expected = sortKVW(jsInnerJoin(leftRecs, rightRecs));
              expect(actual).toEqual(expected);
            } finally {
              out.dispose();
            }
          } finally {
            left.dispose();
            right.dispose();
          }
        },
      ),
      { numRuns: 200 },
    );
  });

  it('random string tables: outer join row count matches oracle', () => {
    const pool = ['x', 'y', 'z', null];

    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(...pool), { minLength: 1, maxLength: 15 }),
        fc.array(fc.constantFrom(...pool), { minLength: 0, maxLength: 15 }),
        (leftKeys, rightKeys) => {
          const left  = makeDF(rt, { k: leftKeys  }, { k: 'utf8' });
          const right = makeDF(rt, { k: rightKeys }, { k: 'utf8' });
          try {
            const out = left.join(right, { on: 'k', how: 'left' });
            try {
              // Every left row must appear at least once.
              expect(out.shape[0]).toBeGreaterThanOrEqual(leftKeys.length);
              // Every left row produces at least one output row.
              const outRecs = out.toRecords();
              expect(outRecs.length).toBeGreaterThanOrEqual(leftKeys.length);
            } finally {
              out.dispose();
            }
          } finally {
            left.dispose();
            right.dispose();
          }
        },
      ),
      { numRuns: 100 },
    );
  });
});
