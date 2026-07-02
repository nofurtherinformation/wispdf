/**
 * Property tests (fast-check): compiled-plan results vs the naive JS oracle over
 * random small frames, for every op incl. nulls / NaN / ±inf / zero divisors.
 * Both consume the same `resolve()` output, isolating the compiler's kernel lowering.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import fc from 'fast-check';
import { col, lit, type Expr } from '../../src/expr/ast.js';
import { compile } from '../../src/expr/compile.js';
import { resolve, schemaOf } from '../../src/expr/dtypes.js';
import { columnToArray, freeColumn } from '../../src/memory/column.js';
import type { DType } from '../../src/memory/dtype.js';
import { loadEnv, TestFrame, type TestEnv } from './helper.js';
import { naiveColumn, naiveScalar, cellEq, cellClose, type Cell, type JSFrame } from './naive.js';

let env: TestEnv;
beforeAll(async () => { env = await loadEnv(); });

const SCHEMA = schemaOf({
  a: 'f64', b: 'f64', c: 'i32', d: 'i32', e: 'f32', g: 'u32', s: 'utf8', k: 'bool',
});
const DTYPES: Record<string, DType> = {
  a: 'f64', b: 'f64', c: 'i32', d: 'i32', e: 'f32', g: 'u32', s: 'utf8', k: 'bool',
};

// f32-exact discrete floats keep f32 element-wise ops bit-identical to the oracle.
const floatVals = [0, 1, -1, 2, -2, 3, 0.5, -0.5, 10, -10, NaN, Infinity, -Infinity];
const intVals = [-30, -3, -1, 0, 1, 2, 3, 7, 30];
const uintVals = [0, 1, 2, 3, 7, 30, 60];
const strVals = ['x', 'y', 'z', 'a', ''];

function cellArb<T>(vals: readonly T[]): fc.Arbitrary<T | null> {
  return fc.oneof(
    { arbitrary: fc.constant(null), weight: 1 },
    { arbitrary: fc.constantFrom(...vals), weight: 5 },
  );
}
const ARB: Record<string, fc.Arbitrary<Cell>> = {
  a: cellArb(floatVals), b: cellArb(floatVals), e: cellArb(floatVals),
  c: cellArb(intVals), d: cellArb(intVals), g: cellArb(uintVals),
  s: cellArb(strVals), k: cellArb([true, false]),
};

const frameArb = fc.integer({ min: 0, max: 8 }).chain((n) =>
  fc
    .record(Object.fromEntries(Object.keys(ARB).map((k) => [k, fc.array(ARB[k]!, { minLength: n, maxLength: n })])))
    .map((cols) => ({ n, cols: cols as JSFrame })),
);

function specsFrom(cols: JSFrame): Record<string, { dtype: DType; values: Cell[] }> {
  const out: Record<string, { dtype: DType; values: Cell[] }> = {};
  for (const [name, values] of Object.entries(cols)) out[name] = { dtype: DTYPES[name]!, values };
  return out;
}

function compare(got: Cell[], want: Cell[], close: boolean): void {
  expect(got.length, 'length').toBe(want.length);
  for (let i = 0; i < got.length; i++) {
    const ok = close ? cellClose(got[i]!, want[i]!) : cellEq(got[i]!, want[i]!);
    if (!ok) throw new Error(`mismatch at [${i}]: got ${String(got[i])} want ${String(want[i])}`);
  }
}

// ── Element-wise ──────────────────────────────────────────────────────────────

const ELEMENTWISE: Expr[] = [
  // arithmetic (same dtype)
  col('a').add(col('b')), col('a').sub(col('b')), col('a').mul(col('b')),
  col('a').div(col('b')), col('a').mod(col('b')),
  col('c').add(col('d')), col('c').sub(col('d')), col('c').mul(col('d')),
  col('c').div(col('d')), col('c').mod(col('d')),
  col('g').add(col('g')), col('g').sub(col('g')), col('g').mul(col('g')),
  col('e').add(col('e')), col('e').mul(col('e')), col('e').div(col('e')),
  // scalar arithmetic + neg + left scalar
  col('a').add(5), col('a').mul(2), col('c').add(2), col('a').neg(), col('c').neg(),
  lit(10).sub(col('a')), lit(12).div(col('c')),
  // widening
  col('c').add(col('a')), col('c').mul(col('e')),
  // comparisons
  col('a').gt(col('b')), col('a').ge(col('b')), col('a').lt(col('b')),
  col('a').le(col('b')), col('a').eq(col('b')), col('a').ne(col('b')),
  col('c').gt(col('d')), col('c').eq(col('d')),
  col('a').gt(0), col('a').le(1), col('c').ne(0),
  // boolean Kleene
  col('a').gt(0).and(col('b').gt(0)), col('a').gt(0).or(col('b').gt(0)),
  col('a').gt(0).not(), col('k').and(col('a').gt(0)), col('k').or(col('b').lt(0)),
  // null utilities
  col('a').isNull(), col('a').notNull(), col('a').fillNull(0), col('c').fillNull(-1),
  col('k').fillNull(false), col('s').fillNull('a'), col('s').fillNull('NEW'),
  // casts
  col('a').cast('i32'), col('a').cast('u32'), col('a').cast('f32'), col('a').cast('bool'),
  col('c').cast('f64'), col('c').cast('f32'), col('c').cast('u32'), col('c').cast('bool'),
  col('g').cast('i32'), col('e').cast('f64'), col('e').cast('i32'), col('k').cast('i32'),
  // string equality
  col('s').eq('x'), col('s').ne('y'), col('s').eq('NOPE'),
  // aggregate broadcast into element-wise
  col('a').sub(col('a').mean()), col('c').sub(col('c').max()),
];

describe('element-wise ops match the naive oracle', () => {
  it('holds over random frames', () => {
    fc.assert(
      fc.property(frameArb, ({ n, cols }) => {
        const f = new TestFrame(env, specsFrom(cols));
        try {
          for (const expr of ELEMENTWISE) {
            const t = resolve(expr, SCHEMA);
            const p = compile(expr, f).execute();
            const got = columnToArray(f.ctx, p.column!);
            freeColumn(f.ctx, p.column!);
            const want = naiveColumn(t, cols, n);
            // aggregate-broadcast sub uses float means → allow tiny slack
            const close = t.kind === 'arith' && t.dtype === 'f64';
            try {
              compare(got, want, close);
            } catch (err) {
              throw new Error(`${expr}: ${(err as Error).message}`);
            }
          }
        } finally {
          f.free();
        }
        return true;
      }),
      { numRuns: 60 },
    );
  });
});

// ── Aggregations ──────────────────────────────────────────────────────────────

const AGG_EXACT: Expr[] = [
  col('a').min(), col('a').max(), col('a').count(), col('a').nunique(),
  col('a').first(), col('a').last(),
  col('c').min(), col('c').max(), col('c').count(), col('c').nunique(),
  col('c').first(), col('c').last(),
  col('g').count(), col('g').nunique(),
  col('s').count(), col('s').nunique(), col('s').first(), col('s').last(),
];
const AGG_CLOSE: Expr[] = [
  col('a').sum(), col('a').mean(), col('a').std(), col('a').var(),
  col('c').sum(), col('c').mean(), col('c').std(), col('c').var(),
  col('g').sum(),
];

describe('aggregations match the naive oracle', () => {
  it('exact reductions (min/max/count/nunique/first/last)', () => {
    fc.assert(
      fc.property(frameArb, ({ n, cols }) => {
        const f = new TestFrame(env, specsFrom(cols));
        try {
          for (const expr of AGG_EXACT) {
            const t = resolve(expr, SCHEMA);
            const got = compile(expr, f).execute().scalar!.value;
            const want = naiveScalar(t as never, cols, n);
            if (!cellEq(got, want)) throw new Error(`${expr}: got ${String(got)} want ${String(want)}`);
          }
        } finally {
          f.free();
        }
        return true;
      }),
      { numRuns: 60 },
    );
  });

  it('float reductions (sum/mean/std/var), tolerant of accumulation order', () => {
    fc.assert(
      fc.property(frameArb, ({ n, cols }) => {
        const f = new TestFrame(env, specsFrom(cols));
        try {
          for (const expr of AGG_CLOSE) {
            const t = resolve(expr, SCHEMA);
            const got = compile(expr, f).execute().scalar!.value;
            const want = naiveScalar(t as never, cols, n);
            if (!cellClose(got, want, 1e-6)) throw new Error(`${expr}: got ${String(got)} want ${String(want)}`);
          }
        } finally {
          f.free();
        }
        return true;
      }),
      { numRuns: 60 },
    );
  });
});

// ── Determinism across builds (ADR-004) ───────────────────────────────────────

describe('scalar and SIMD builds agree', () => {
  it('produce identical results for a sample of expressions', async () => {
    const simd = await loadEnv(true);
    const specs = {
      a: { dtype: 'f64' as const, values: [1, null, 3.5, NaN, 5] },
      b: { dtype: 'f64' as const, values: [2, 2, 2, 2, 0] },
      c: { dtype: 'i32' as const, values: [4, 0, -3, 7, 2] },
      s: { dtype: 'utf8' as const, values: ['x', 'y', null, 'x', 'z'] },
    };
    const fS = new TestFrame(env, specs);
    const fV = new TestFrame(simd, specs);
    try {
      for (const expr of [
        col('a').add(col('b')), col('a').div(col('b')), col('c').div(col('c')),
        col('a').gt(2), col('a').gt(0).and(col('b').lt(3)), col('a').cast('i32'),
        col('s').eq('x'), col('a').fillNull(-1),
      ]) {
        const rs = compile(expr, fS).execute();
        const rv = compile(expr, fV).execute();
        const as = columnToArray(fS.ctx, rs.column!);
        const av = columnToArray(fV.ctx, rv.column!);
        freeColumn(fS.ctx, rs.column!);
        freeColumn(fV.ctx, rv.column!);
        for (let i = 0; i < as.length; i++) {
          expect(cellEq(as[i]!, av[i]!), `${expr}[${i}]`).toBe(true);
        }
      }
    } finally {
      fS.free();
      fV.free();
    }
  });
});
