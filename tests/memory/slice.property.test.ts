import { describe, it, expect, beforeAll } from 'vitest';
import fc from 'fast-check';
import { ctxForTest } from './helper.js';
import type { MemoryContext } from '../../src/memory/context.js';
import {
  createColumn,
  columnToArray,
  sliceColumn,
  freeColumn,
  DTYPES,
  decodeStats,
  type Cell,
} from '../../src/memory/index.js';

// P1.2 zero-copy slice gate (deliverable §4): a slice shares the parent's buffers
// with a data byte-offset baked into dataPtr and a validity BIT offset recorded in
// validityBitOffset. Correctness incl. slice-of-slice and non-byte-aligned starts.

let ctx: MemoryContext;
beforeAll(async () => {
  ctx = await ctxForTest(false);
});

function cellEq(a: Cell, b: Cell): boolean {
  if (a === null || b === null) return a === b;
  if (typeof a === 'number' && typeof b === 'number') return Object.is(a, b);
  return a === b;
}

function expectArrayEq(got: Cell[], want: Cell[]): void {
  expect(got.length).toBe(want.length);
  for (let i = 0; i < want.length; i++) {
    if (!cellEq(got[i]!, want[i]!)) {
      throw new Error(`i=${i} got=${String(got[i])} want=${String(want[i])}`);
    }
  }
}

const f64arb = fc.oneof(fc.constant(null), fc.constantFrom(NaN, Infinity, 0, -0), fc.double({ noNaN: true, noDefaultInfinity: true }));
const strArb = fc.oneof(fc.constant(null), fc.string(), fc.constantFrom('a', 'a', 'b', '😀', ''));

describe('zero-copy slice shares buffers (no copy)', () => {
  it('bakes the data offset into dataPtr and records a validity bit offset', () => {
    const input: (number | null)[] = [0, 1, null, 3, 4, null, 6, 7, 8, 9];
    const col = createColumn(ctx, 'f64', input);
    try {
      const sl = sliceColumn(col, 3, 8); // elements 3..7
      expect(sl.owned).toBe(false);
      expect(sl.length).toBe(5);
      expect(sl.dataPtr).toBe(col.dataPtr + 3 * DTYPES.f64.size);
      expect(sl.validityPtr).toBe(col.validityPtr); // same shared bitmap
      expect(sl.validityBitOffset).toBe(3);
      expectArrayEq(columnToArray(ctx, sl), input.slice(3, 8));

      // Freeing a slice is a no-op: the parent is still fully readable afterwards.
      freeColumn(ctx, sl);
      expectArrayEq(columnToArray(ctx, col), input);
    } finally {
      freeColumn(ctx, col);
    }
  });

  it('a utf8 slice shares the parent dictionary and its decode cache', () => {
    const input = ['a', 'b', 'a', null, 'c', 'b', 'a'];
    const col = createColumn(ctx, 'utf8', input);
    try {
      const sl = sliceColumn(col, 2, 6);
      expect(sl.dict).toBe(col.dict); // same dictionary identity
      // Decoding the parent warms the shared cache; the slice adds only hits.
      columnToArray(ctx, col);
      const before = decodeStats(col.dict!);
      columnToArray(ctx, sl);
      const after = decodeStats(col.dict!);
      expect(after.misses).toBe(before.misses); // no new boundary crossings
      expectArrayEq(columnToArray(ctx, sl), input.slice(2, 6));
    } finally {
      freeColumn(ctx, col);
    }
  });
});

describe('slice correctness (property, incl. non-8-aligned starts)', () => {
  it('f64: slice(start,end) == input.slice(start,end) for random bounds', () => {
    fc.assert(
      fc.property(
        fc.array(f64arb, { minLength: 1, maxLength: 40 }),
        fc.nat(45),
        fc.nat(45),
        (input, a, b) => {
          const start = Math.min(a, b);
          const end = Math.max(a, b);
          const col = createColumn(ctx, 'f64', input);
          try {
            const sl = sliceColumn(col, start, end);
            expectArrayEq(columnToArray(ctx, sl), input.slice(start, end) as Cell[]);
          } finally {
            freeColumn(ctx, col);
          }
        },
      ),
      { numRuns: 250 },
    );
  });

  it('utf8: slice(start,end) == input.slice(start,end) for random bounds', () => {
    fc.assert(
      fc.property(
        fc.array(strArb, { minLength: 1, maxLength: 40 }),
        fc.nat(45),
        fc.nat(45),
        (input, a, b) => {
          const start = Math.min(a, b);
          const end = Math.max(a, b);
          const col = createColumn(ctx, 'utf8', input);
          try {
            const sl = sliceColumn(col, start, end);
            expectArrayEq(columnToArray(ctx, sl), input.slice(start, end) as Cell[]);
          } finally {
            freeColumn(ctx, col);
          }
        },
      ),
      { numRuns: 200 },
    );
  });
});

describe('slice-of-slice composes (offsets accumulate)', () => {
  it('nested slices match nested Array.slice for f64 and utf8', () => {
    fc.assert(
      fc.property(
        fc.array(f64arb, { minLength: 1, maxLength: 40 }),
        fc.nat(45),
        fc.nat(45),
        fc.nat(45),
        fc.nat(45),
        (input, a, b, c, d) => {
          const s1 = Math.min(a, b);
          const e1 = Math.max(a, b);
          const s2 = Math.min(c, d);
          const e2 = Math.max(c, d);
          const col = createColumn(ctx, 'f64', input);
          try {
            const sl1 = sliceColumn(col, s1, e1);
            const sl2 = sliceColumn(sl1, s2, e2);
            const want = input.slice(s1, e1).slice(s2, e2) as Cell[];
            // pointer + bit-offset accumulation
            expect(sl2.validityBitOffset).toBe(sl1.validityBitOffset + Math.min(s2, sl1.length));
            expectArrayEq(columnToArray(ctx, sl2), want);
          } finally {
            freeColumn(ctx, col);
          }
        },
      ),
      { numRuns: 250 },
    );
  });

  it('empty slice and out-of-range bounds clamp to []', () => {
    const col = createColumn(ctx, 'i32', [1, 2, 3, 4, 5]);
    try {
      expect(columnToArray(ctx, sliceColumn(col, 3, 3))).toEqual([]);
      expect(columnToArray(ctx, sliceColumn(col, 4, 2))).toEqual([]); // end < start
      expect(columnToArray(ctx, sliceColumn(col, 10, 20))).toEqual([]); // past end
      expect(columnToArray(ctx, sliceColumn(col, -5, 2))).toEqual([1, 2]); // start clamps to 0
    } finally {
      freeColumn(ctx, col);
    }
  });
});
