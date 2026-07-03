/**
 * i64 column construction: conformance frame_error cases (fixtures/i64.json)
 * plus round-trip property tests.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import fc from 'fast-check';
import { ctxForTest } from './helper.js';
import type { MemoryContext } from '../../src/memory/context.js';
import { createColumn, columnToArray, freeColumn } from '../../src/memory/index.js';

let ctx: MemoryContext;
beforeAll(async () => {
  ctx = await ctxForTest(false);
});

// ---- conformance frame_error cases from fixtures/i64.json ----

describe('i64 column construction errors (ADR-009)', () => {
  it('construct_i64__unsafe_int_positive_throws: 9007199254740992 (2^53) throws', () => {
    expect(() => createColumn(ctx, 'i64', [9007199254740992])).toThrow('9007199254740992');
  });

  it('construct_i64__unsafe_int_negative_throws: -9007199254740992 (-2^53) throws', () => {
    expect(() => createColumn(ctx, 'i64', [-9007199254740992])).toThrow('9007199254740992');
  });

  it('construct_i64__non_integer_throws: 1.5 throws', () => {
    expect(() => createColumn(ctx, 'i64', [1.5])).toThrow('1.5');
  });

  it('construct_i64__nan_throws: NaN throws', () => {
    expect(() => createColumn(ctx, 'i64', [NaN])).toThrow('NaN');
  });
});

// ---- round-trip property tests ----

describe('i64 column round-trip', () => {
  const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);
  const i64arb = fc.oneof(
    fc.constant(null),
    fc.bigInt({ min: -MAX_SAFE, max: MAX_SAFE }),
  );

  it('BigInt64Array fast path: bulk-copied and read back exactly', () => {
    fc.assert(
      fc.property(fc.array(fc.bigInt({ min: -MAX_SAFE, max: MAX_SAFE }), { maxLength: 40 }), (vals) => {
        const input = BigInt64Array.from(vals);
        const col = createColumn(ctx, 'i64', input);
        try {
          expect(col.validityPtr).toBe(0);
          const out = columnToArray(ctx, col);
          expect(out).toEqual(Array.from(input));
        } finally {
          freeColumn(ctx, col);
        }
      }),
      { numRuns: 150 },
    );
  });

  it('slow path (mixed bigint | null): preserves values and nulls', () => {
    fc.assert(
      fc.property(fc.array(i64arb, { maxLength: 40 }), (input) => {
        const col = createColumn(ctx, 'i64', input as never);
        try {
          const hasNull = input.some((v) => v === null);
          expect(col.validityPtr === 0).toBe(!hasNull);
          const out = columnToArray(ctx, col);
          expect(out.length).toBe(input.length);
          for (let i = 0; i < input.length; i++) {
            expect(out[i]).toBe(input[i] === null ? null : BigInt(input[i] as bigint));
          }
        } finally {
          freeColumn(ctx, col);
        }
      }),
      { numRuns: 200 },
    );
  });

  it('safe integer numbers are accepted and yield bigint output', () => {
    const col = createColumn(ctx, 'i64', [0, 1, -1, Number.MAX_SAFE_INTEGER, Number.MIN_SAFE_INTEGER, null]);
    try {
      expect(columnToArray(ctx, col)).toEqual([0n, 1n, -1n, BigInt(Number.MAX_SAFE_INTEGER), BigInt(Number.MIN_SAFE_INTEGER), null]);
    } finally {
      freeColumn(ctx, col);
    }
  });

  it('empty i64 column round-trips to []', () => {
    const col = createColumn(ctx, 'i64', []);
    try {
      expect(col.length).toBe(0);
      expect(col.validityPtr).toBe(0);
      expect(columnToArray(ctx, col)).toEqual([]);
    } finally {
      freeColumn(ctx, col);
    }
  });

  it('all-null i64 column round-trips to all null', () => {
    const col = createColumn(ctx, 'i64', [null, null, null]);
    try {
      expect(col.validityPtr).not.toBe(0);
      expect(columnToArray(ctx, col)).toEqual([null, null, null]);
    } finally {
      freeColumn(ctx, col);
    }
  });

  it('INT64 boundary values survive round-trip', () => {
    const INT64_MAX = 9223372036854775807n;
    const INT64_MIN = -9223372036854775808n;
    const col = createColumn(ctx, 'i64', BigInt64Array.from([INT64_MAX, INT64_MIN, 0n]));
    try {
      expect(columnToArray(ctx, col)).toEqual([INT64_MAX, INT64_MIN, 0n]);
    } finally {
      freeColumn(ctx, col);
    }
  });
});
