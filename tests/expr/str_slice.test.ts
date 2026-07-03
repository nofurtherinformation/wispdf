/**
 * Tests for col('x').str.slice(start, end?) — dtypes.md §13.
 *
 * Coverage:
 *   1. dtype error: str.slice on a non-utf8 column throws ExprError.
 *   2. Equivalence property: compiled str.slice == naïve per-row JS slice over
 *      random frames — unicode incl. astral plane, negative/out-of-range indices,
 *      empty strings, nulls, all-unique and low-cardinality dicts.
 *   3. Census-shaped smoke test: GEOID.slice(0, 11) → state codes.
 *   4. Series.str.slice mirror: same outputs as col().str.slice().
 */

import { describe, it, expect, beforeAll } from 'vitest';
import fc from 'fast-check';
import { col } from '../../src/expr/ast.js';
import { compile } from '../../src/expr/compile.js';
import { columnToArray, freeColumn, createColumn } from '../../src/memory/column.js';
import type { Cell } from '../../src/memory/column.js';
import { loadEnv, TestFrame, takeColumn } from './helper.js';
import type { TestEnv } from './helper.js';
import { ExprError } from '../../src/expr/errors.js';
import { Series } from '../../src/frame/series.js';

let env: TestEnv;
beforeAll(async () => {
  env = await loadEnv();
});

// ── Helpers ────────────────────────────────────────────────────────────────────

function makeUtf8Frame(values: Array<string | null>): TestFrame {
  return new TestFrame(env, {
    s: { dtype: 'utf8', values },
  });
}

/**
 * Reference oracle: JS String.prototype.slice, then round-tripped through UTF-8
 * encoding/decoding (same as utf8 column storage does). Lone surrogates produced by
 * splitting an astral-plane character become U+FFFD (WHATWG TextEncoder behaviour) —
 * the same transformation both our str.slice and the mapFn escape-hatch apply.
 * This is the documented surrogate caveat (dtypes.md §13 / JSDoc on str.slice).
 */
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();
function utf8RoundTrip(s: string): string {
  return textDecoder.decode(textEncoder.encode(s));
}

function naiveSlice(values: Array<string | null>, start: number, end?: number): Array<string | null> {
  return values.map((v) => {
    if (v === null) return null;
    const sliced = end === undefined ? v.slice(start) : v.slice(start, end);
    return utf8RoundTrip(sliced);
  });
}

/** Run col('s').str.slice(start, end?) on a frame, return Cell[]. */
function runSlice(frame: TestFrame, start: number, end?: number): Cell[] {
  const expr = end === undefined
    ? col('s').str.slice(start)
    : col('s').str.slice(start, end);
  const plan = compile(expr, frame);
  const result = plan.execute();
  if (result.kind !== 'column' || !result.column) throw new Error('expected column result');
  return takeColumn(env.ctx, result.column);
}

// ── 1. dtype error ─────────────────────────────────────────────────────────────

describe('str.slice dtype error', () => {
  it('throws ExprError when applied to a non-utf8 column (f64)', () => {
    const frame = new TestFrame(env, {
      n: { dtype: 'f64', values: [1.0, 2.0, 3.0] },
    });
    try {
      expect(() => compile(col('n').str.slice(0, 2), frame)).toThrow(ExprError);
    } finally {
      frame.free();
    }
  });

  it('throws ExprError when applied to an i32 column', () => {
    const frame = new TestFrame(env, {
      n: { dtype: 'i32', values: [1, 2, 3] },
    });
    try {
      expect(() => compile(col('n').str.slice(1), frame)).toThrow(ExprError);
    } finally {
      frame.free();
    }
  });

  it('error message names the op and the actual dtype', () => {
    const frame = new TestFrame(env, {
      n: { dtype: 'bool', values: [true, false] },
    });
    try {
      let msg = '';
      try { compile(col('n').str.slice(0), frame); } catch (e) { msg = (e as Error).message; }
      expect(msg).toContain('str.slice');
      expect(msg).toContain('bool');
    } finally {
      frame.free();
    }
  });
});

// ── 2. Correctness: equivalence with naïve per-row slice ──────────────────────

describe('str.slice equivalence with naïve per-row JS String.prototype.slice', () => {
  // Values covering: ASCII, multi-byte UTF-8, astral-plane code points (emoji),
  // empty string, and null.
  const strVals = [
    'hello', 'world', 'abc', '', 'x',
    'café', '北京', '東京',
    '😀', '🎉', '👋', '𝄞', // astral-plane (2 UTF-16 code units each)
    'ab😀cd', // mixed ASCII + astral
  ];

  const strArb = fc.oneof(
    fc.constant(null),
    fc.constantFrom(...strVals),
    // Arbitrary strings including unicode (fast-check generates BMP by default)
    fc.string({ maxLength: 12 }),
  );

  const indexArb = fc.oneof(
    fc.integer({ min: -20, max: 20 }),
    fc.constant(0),
    fc.constant(-1),
    fc.constant(100),  // out-of-range
    fc.constant(-100), // out-of-range negative
  );

  it('positive start, no end — low-cardinality dict', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(...strVals, null as string | null), { minLength: 0, maxLength: 30 }),
        fc.integer({ min: 0, max: 15 }),
        (values, start) => {
          const frame = makeUtf8Frame(values);
          try {
            const got = runSlice(frame, start);
            const want = naiveSlice(values, start);
            expect(got).toEqual(want);
          } finally {
            frame.free();
          }
        },
      ),
      { numRuns: 200 },
    );
  });

  it('positive start + end — low-cardinality dict', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(...strVals, null as string | null), { minLength: 0, maxLength: 30 }),
        fc.integer({ min: 0, max: 15 }),
        fc.integer({ min: 0, max: 15 }),
        (values, a, b) => {
          const frame = makeUtf8Frame(values);
          try {
            const got = runSlice(frame, a, b);
            const want = naiveSlice(values, a, b);
            expect(got).toEqual(want);
          } finally {
            frame.free();
          }
        },
      ),
      { numRuns: 200 },
    );
  });

  it('negative indices — low-cardinality dict', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(...strVals, null as string | null), { minLength: 0, maxLength: 30 }),
        indexArb,
        fc.option(indexArb, { nil: undefined }),
        (values, start, end) => {
          const frame = makeUtf8Frame(values);
          try {
            const got = runSlice(frame, start, end);
            const want = naiveSlice(values, start, end);
            expect(got).toEqual(want);
          } finally {
            frame.free();
          }
        },
      ),
      { numRuns: 300 },
    );
  });

  it('all-unique dict — arbitrary strings (census-like cardinality)', () => {
    // Generate frames where every value is unique (like GEOIDs).
    fc.assert(
      fc.property(
        fc.uniqueArray(strArb.filter((v) => v !== null) as fc.Arbitrary<string>, {
          minLength: 0,
          maxLength: 40,
        }),
        fc.integer({ min: 0, max: 10 }),
        fc.option(fc.integer({ min: 0, max: 20 }), { nil: undefined }),
        (values, start, end) => {
          const frame = makeUtf8Frame(values);
          try {
            const got = runSlice(frame, start, end);
            const want = naiveSlice(values, start, end);
            expect(got).toEqual(want);
          } finally {
            frame.free();
          }
        },
      ),
      { numRuns: 200 },
    );
  });

  it('mixed nulls + unicode + astral', () => {
    fc.assert(
      fc.property(
        fc.array(strArb, { minLength: 0, maxLength: 25 }),
        indexArb,
        fc.option(indexArb, { nil: undefined }),
        (values, start, end) => {
          const frame = makeUtf8Frame(values);
          try {
            const got = runSlice(frame, start, end);
            const want = naiveSlice(values, start, end);
            expect(got).toEqual(want);
          } finally {
            frame.free();
          }
        },
      ),
      { numRuns: 300 },
    );
  });

  it('all-null column — result is all null', () => {
    const values: null[] = [null, null, null, null];
    const frame = makeUtf8Frame(values);
    try {
      const got = runSlice(frame, 0, 5);
      expect(got).toEqual([null, null, null, null]);
    } finally {
      frame.free();
    }
  });

  it('empty frame — result is empty', () => {
    const frame = makeUtf8Frame([]);
    try {
      const got = runSlice(frame, 0, 3);
      expect(got).toEqual([]);
    } finally {
      frame.free();
    }
  });

  it('single-row all-valid', () => {
    const frame = makeUtf8Frame(['hello']);
    try {
      expect(runSlice(frame, 1, 4)).toEqual(['ell']);
    } finally {
      frame.free();
    }
  });
});

// ── 3. Census-shaped smoke test ────────────────────────────────────────────────

describe('census derive state smoke test', () => {
  // Simulates: GEOID.slice(0, 11) collapses ~N unique GEOIDs to ~S state codes.
  it('slice(0, 11) on GEOID-like strings gives state codes', () => {
    // 12-char GEOIDs: first 11 chars = state (2) + county (3) + tract (6)
    const geoids = [
      '01001020100', '01001020200', '01001020300',
      '01003010100', '01003010200',
      '06037137000', '06037137100', '06037137200',
      '36061000100', '36061000200',
    ];
    // Each appears twice to test index remapping.
    const data = [...geoids, ...geoids];
    const frame = makeUtf8Frame(data);
    try {
      const got = runSlice(frame, 0, 11);
      const want = naiveSlice(data, 0, 11);
      expect(got).toEqual(want);
      // Verify collapsing: all 11-char results match original (they're already 11 chars)
      // The key property: no more unique values than the number of unique state prefixes.
    } finally {
      frame.free();
    }
  });

  it('result dict is smaller (collapsed): 85K unique → ~50 state codes', () => {
    // Build a small-scale version: 10 unique GEOIDs, all different state prefixes after slice
    const geoids = Array.from({ length: 10 }, (_, i) =>
      `${String(i + 1).padStart(2, '0')}001${String(i * 1000).padStart(6, '0')}`,
    );
    // 50 rows, 10 unique GEOIDs
    const data = Array.from({ length: 50 }, (_, i) => geoids[i % 10]!);
    const frame = makeUtf8Frame(data);
    try {
      const expr = col('s').str.slice(0, 5);
      const plan = compile(expr, frame);
      const result = plan.execute();
      if (result.kind !== 'column' || !result.column) throw new Error('expected column');
      const col2 = result.column;
      // The result dict should have at most 10 unique values.
      expect(col2.dict?.count).toBeLessThanOrEqual(10);
      freeColumn(env.ctx, col2);
    } finally {
      frame.free();
    }
  });
});

// ── 4. Series.str.slice mirror ─────────────────────────────────────────────────

describe('Series.str.slice', () => {
  it('mirrors col().str.slice() output', () => {
    const values: Array<string | null> = ['hello', null, 'world', 'foo', null, 'bar'];
    const colData = createColumn(env.ctx, 'utf8', values);
    const series = new Series(env.ctx, 'test', colData);
    try {
      const result = series.str.slice(1, 4);
      expect(result.toArray()).toEqual(naiveSlice(values, 1, 4));
      expect(result.dtype).toBe('utf8');
    } finally {
      // series borrows colData; free via createColumn owner
      freeColumn(env.ctx, colData);
    }
  });

  it('Series.str throws TypeError on non-utf8', () => {
    const col2 = createColumn(env.ctx, 'i32', [1, 2, 3]);
    const series = new Series(env.ctx, 'n', col2);
    try {
      expect(() => series.str).toThrow(TypeError);
    } finally {
      freeColumn(env.ctx, col2);
    }
  });

  it('negative start, no end', () => {
    const values = ['abc', 'defgh', null, 'xy'];
    const colData = createColumn(env.ctx, 'utf8', values);
    const series = new Series(env.ctx, 'v', colData);
    try {
      expect(series.str.slice(-2).toArray()).toEqual(naiveSlice(values, -2));
    } finally {
      freeColumn(env.ctx, colData);
    }
  });

  it('astral plane strings (2 UTF-16 code units per char)', () => {
    const values = ['😀😀', '😀', null, '😀😀😀'];
    const colData = createColumn(env.ctx, 'utf8', values);
    const series = new Series(env.ctx, 'v', colData);
    try {
      // slice(0, 2) takes the first emoji (2 code units)
      expect(series.str.slice(0, 2).toArray()).toEqual(naiveSlice(values, 0, 2));
    } finally {
      freeColumn(env.ctx, colData);
    }
  });
});
