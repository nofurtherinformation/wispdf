import { describe, it, expect, beforeAll } from 'vitest';
import fc from 'fast-check';
import { ctxForTest } from './helper.js';
import type { MemoryContext } from '../../src/memory/context.js';
import {
  createColumn,
  columnToArray,
  freeColumn,
  type Cell,
  type DType,
} from '../../src/memory/index.js';

// P1.2 gate: round-trip JS -> column -> JS for EVERY dtype, with random values
// including nulls, NaN/±inf (f64/f32), empty arrays, length-1, and
// non-multiple-of-8 lengths (validity-bitmap padding). NaN is a VALUE, not a null
// (dtypes.md §4): only `null`/`undefined` become nulls.

let ctx: MemoryContext;
beforeAll(async () => {
  ctx = await ctxForTest(false);
});

/** Equality that treats `null` distinctly and uses `Object.is` for numbers (NaN/-0). */
function cellEq(a: Cell, b: Cell): boolean {
  if (a === null || b === null) return a === b;
  if (typeof a === 'number' && typeof b === 'number') return Object.is(a, b);
  return a === b;
}

/** Assert a full round-trip of `input` under `dtype`, mapping each cell via `expect`. */
function expectRoundTrip(
  dtype: DType,
  input: ReadonlyArray<number | boolean | string | null>,
  expected: (x: number | boolean | string | null) => Cell,
): void {
  const col = createColumn(ctx, dtype, input as never);
  try {
    // All-valid shortcut (ABI §4.1): validityPtr is 0 iff there are no nulls.
    const hasNull = input.some((v) => v === null);
    expect(col.validityPtr === 0).toBe(!hasNull);

    const out = columnToArray(ctx, col);
    expect(out.length).toBe(input.length);
    for (let i = 0; i < input.length; i++) {
      const want = expected(input[i]!);
      if (!cellEq(out[i]!, want)) {
        throw new Error(
          `dtype=${dtype} i=${i} got=${String(out[i])} want=${String(want)} (len=${input.length})`,
        );
      }
    }
  } finally {
    freeColumn(ctx, col);
  }
}

const id = (x: number | boolean | string | null): Cell => x;

// ---- value arbitraries per dtype (nulls interleaved) -----------------------

const f64arb = fc.oneof(
  fc.constant(null),
  fc.constantFrom(NaN, Infinity, -Infinity, 0, -0, 1, -1),
  fc.double({ noNaN: true, noDefaultInfinity: true }),
);
const f32arb = f64arb; // same generator; f32 storage rounds in the expectation
const i32arb = fc.oneof(fc.constant(null), fc.integer({ min: -0x80000000, max: 0x7fffffff }));
const u32arb = fc.oneof(fc.constant(null), fc.integer({ min: 0, max: 0xffffffff }));
const boolarb = fc.oneof(fc.constant(null), fc.boolean());
const strarb = fc.oneof(
  fc.constant(null),
  fc.string(),
  fc.fullUnicodeString(),
  fc.constantFrom('', 'a', 'a', 'b', 'foo', '😀', 'héllo', '你好', 'a'),
);

describe('column round-trip (slow path: plain arrays with nulls)', () => {
  it('f64 preserves values, NaN/±inf, and nulls', () => {
    fc.assert(
      fc.property(fc.array(f64arb, { maxLength: 40 }), (input) =>
        expectRoundTrip('f64', input, id),
      ),
      { numRuns: 250 },
    );
  });

  it('f32 preserves f32-rounded values, NaN/±inf, and nulls', () => {
    fc.assert(
      fc.property(fc.array(f32arb, { maxLength: 40 }), (input) =>
        expectRoundTrip('f32', input, (x) => (x === null ? null : Math.fround(x as number))),
      ),
      { numRuns: 250 },
    );
  });

  it('i32 preserves signed 32-bit integers and nulls', () => {
    fc.assert(
      fc.property(fc.array(i32arb, { maxLength: 40 }), (input) =>
        expectRoundTrip('i32', input, id),
      ),
      { numRuns: 200 },
    );
  });

  it('u32 preserves unsigned 32-bit integers and nulls', () => {
    fc.assert(
      fc.property(fc.array(u32arb, { maxLength: 40 }), (input) =>
        expectRoundTrip('u32', input, id),
      ),
      { numRuns: 200 },
    );
  });

  it('bool preserves booleans and nulls', () => {
    fc.assert(
      fc.property(fc.array(boolarb, { maxLength: 40 }), (input) =>
        expectRoundTrip('bool', input, id),
      ),
      { numRuns: 200 },
    );
  });

  it('utf8 preserves strings (incl. unicode, empty, duplicates) and nulls', () => {
    fc.assert(
      fc.property(fc.array(strarb, { maxLength: 40 }), (input) =>
        expectRoundTrip('utf8', input, id),
      ),
      { numRuns: 250 },
    );
  });
});

describe('column round-trip (fast path: matching TypedArray, no nulls)', () => {
  it('Float64Array is bulk-copied and read back exactly', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ noNaN: true, noDefaultInfinity: true }), { maxLength: 40 }), (nums) => {
        const input = Float64Array.from(nums);
        const col = createColumn(ctx, 'f64', input);
        try {
          expect(col.validityPtr).toBe(0);
          const out = columnToArray(ctx, col);
          for (let i = 0; i < input.length; i++) expect(Object.is(out[i], input[i])).toBe(true);
        } finally {
          freeColumn(ctx, col);
        }
      }),
      { numRuns: 150 },
    );
  });

  it('Int32Array / Uint32Array / Float32Array / Uint8Array(bool) fast paths', () => {
    fc.assert(
      fc.property(fc.array(fc.integer({ min: -1000, max: 1000 }), { maxLength: 32 }), (nums) => {
        const i32 = createColumn(ctx, 'i32', Int32Array.from(nums));
        const u32 = createColumn(ctx, 'u32', Uint32Array.from(nums.map((n) => n >>> 0)));
        const f32 = createColumn(ctx, 'f32', Float32Array.from(nums));
        const bl = createColumn(ctx, 'bool', Uint8Array.from(nums.map((n) => (n & 1 ? 1 : 0))));
        try {
          expect(columnToArray(ctx, i32)).toEqual(nums);
          expect(columnToArray(ctx, u32)).toEqual(nums.map((n) => n >>> 0));
          expect(columnToArray(ctx, f32)).toEqual(nums.map((n) => Math.fround(n)));
          expect(columnToArray(ctx, bl)).toEqual(nums.map((n) => (n & 1) === 1));
        } finally {
          for (const c of [i32, u32, f32, bl]) freeColumn(ctx, c);
        }
      }),
      { numRuns: 120 },
    );
  });
});

describe('column edge cases (empty / length-1 / bitmap padding)', () => {
  const boundaryLengths = [0, 1, 2, 7, 8, 9, 15, 16, 17, 23, 24, 25];

  it('handles empty, length-1, and non-multiple-of-8 lengths with edge nulls', () => {
    for (const n of boundaryLengths) {
      // null at both ends + a NaN in the middle; everything else a distinct value.
      const f: (number | null)[] = Array.from({ length: n }, (_, i) => i);
      if (n > 0) f[0] = null;
      if (n > 1) f[n - 1] = null;
      if (n > 2) f[n >> 1] = NaN;
      expectRoundTrip('f64', f, id);

      const s: (string | null)[] = Array.from({ length: n }, (_, i) => `s${i % 3}`);
      if (n > 0) s[0] = null;
      if (n > 2) s[n - 1] = null;
      expectRoundTrip('utf8', s, id);

      const b: (boolean | null)[] = Array.from({ length: n }, (_, i) => i % 2 === 0);
      if (n > 1) b[1] = null;
      expectRoundTrip('bool', b, id);
    }
  });

  it('an all-null column round-trips to all null with a non-zero validity buffer', () => {
    const col = createColumn(ctx, 'f64', [null, null, null, null, null, null, null, null, null]);
    try {
      expect(col.validityPtr).not.toBe(0);
      expect(columnToArray(ctx, col)).toEqual(new Array(9).fill(null));
    } finally {
      freeColumn(ctx, col);
    }
  });

  it('an empty column allocates and round-trips to []', () => {
    for (const dt of ['f64', 'f32', 'i32', 'u32', 'bool', 'utf8'] as const) {
      const col = createColumn(ctx, dt, []);
      try {
        expect(col.length).toBe(0);
        expect(col.validityPtr).toBe(0);
        expect(columnToArray(ctx, col)).toEqual([]);
      } finally {
        freeColumn(ctx, col);
      }
    }
  });
});
