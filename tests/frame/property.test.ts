/**
 * Property tests (fast-check): round-trip `fromRecords → ops → toRecords` against a naive
 * JS reference. Uses i32 columns with occasional nulls so comparisons are exact.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import fc from 'fast-check';
import { loadRuntimeForTest, makeDF } from './helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import type { Cell } from '../../src/memory/column.js';
import { col } from '../../src/expr/index.js';

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

/** An i32-or-null cell. */
const cell = fc.option(fc.integer({ min: -50, max: 50 }), { nil: null });
/** A table: parallel `a`, `g` columns of the same length. */
const table = fc
  .nat({ max: 40 })
  .chain((n) =>
    fc.record({
      a: fc.array(cell, { minLength: n, maxLength: n }),
      g: fc.array(fc.integer({ min: 0, max: 3 }), { minLength: n, maxLength: n }),
    }),
  );

describe('round-trip properties', () => {
  it('fromColumns → toColumns is identity (nulls preserved)', () => {
    fc.assert(
      fc.property(table, ({ a, g }) => {
        const df = makeDF(rt, { a, g }, { a: 'i32', g: 'i32' });
        try {
          expect(df.toColumns()).toEqual({ a, g });
        } finally {
          df.dispose();
        }
      }),
    );
  });

  it('filter matches a JS reference (null predicate dropped)', () => {
    fc.assert(
      fc.property(table, fc.integer({ min: -50, max: 50 }), ({ a, g }, k) => {
        const df = makeDF(rt, { a, g }, { a: 'i32', g: 'i32' });
        try {
          const out = df.filter(col('a').gt(k));
          try {
            const expected: Cell[] = a.filter((v) => v !== null && v > k);
            expect(out.toColumns().a).toEqual(expected);
          } finally {
            out.dispose();
          }
        } finally {
          df.dispose();
        }
      }),
    );
  });

  it('sortValues is a stable, nulls-last sort', () => {
    fc.assert(
      fc.property(table, (t) => {
        const a = t.a;
        const id = a.map((_, i) => i);
        const df = makeDF(rt, { a, id }, { a: 'i32', id: 'i32' });
        try {
          const out = df.sortValues('a');
          try {
            // JS stable reference: nulls last, otherwise by value, ties by original index.
            const ref = id
              .slice()
              .sort((x, y) => {
                const ax = a[x]!;
                const ay = a[y]!;
                if (ax === null && ay === null) return x - y;
                if (ax === null) return 1;
                if (ay === null) return -1;
                return ax === ay ? x - y : ax - ay;
              });
            expect(out.toColumns().id).toEqual(ref);
          } finally {
            out.dispose();
          }
        } finally {
          df.dispose();
        }
      }),
    );
  });

  it('groupby sum matches a JS reference', () => {
    fc.assert(
      fc.property(table, ({ a, g }) => {
        const df = makeDF(rt, { a, g }, { a: 'i32', g: 'i32' });
        try {
          const out = df.groupby('g').agg({ a: 'sum' });
          try {
            const ref = new Map<number, number>();
            for (let i = 0; i < a.length; i++) {
              const key = g[i]!;
              const add = a[i] === null ? 0 : (a[i] as number);
              ref.set(key, (ref.get(key) ?? 0) + add);
            }
            for (const rec of out.toRecords()) {
              expect(rec.a).toBe(ref.get(rec.g as number));
            }
            expect(out.shape[0]).toBe(ref.size);
          } finally {
            out.dispose();
          }
        } finally {
          df.dispose();
        }
      }),
    );
  });
});
