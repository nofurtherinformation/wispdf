/**
 * Execution correctness for the compiler over real Phase-2 kernels: every op incl.
 * nulls / NaN / zero-divisor, casts, string eq, aggregations, and the filter path.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { col, lit, type Expr } from '../../src/expr/ast.js';
import { compile, compileFilter } from '../../src/expr/compile.js';
import { freeColumn, type Cell } from '../../src/memory/column.js';
import { loadEnv, TestFrame, takeColumn, type TestEnv, type ColSpec } from './helper.js';

let env: TestEnv;
beforeAll(async () => { env = await loadEnv(); });

function withFrame<T>(specs: Record<string, ColSpec>, fn: (f: TestFrame) => T): T {
  const f = new TestFrame(env, specs);
  try { return fn(f); } finally { f.free(); }
}
function runCol(f: TestFrame, expr: Expr): Cell[] {
  const p = compile(expr, f).execute();
  expect(p.kind, `${expr}`).toBe('column');
  return takeColumn(f.ctx, p.column!);
}
function runScalar(f: TestFrame, expr: Expr): Cell {
  const p = compile(expr, f).execute();
  expect(p.kind, `${expr}`).toBe('scalar');
  return p.scalar!.value;
}

describe('arithmetic', () => {
  it('scalar add, chained', () => {
    withFrame({ a: { dtype: 'f64', values: [1, 2, 3] } }, (f) => {
      expect(runCol(f, col('a').add(10))).toEqual([11, 12, 13]);
      expect(runCol(f, col('a').add(1).mul(2).sub(3))).toEqual([1, 3, 5]);
    });
  });
  it('vector add/sub/mul', () => {
    withFrame({ a: { dtype: 'f64', values: [1, 2, 3] }, b: { dtype: 'f64', values: [10, 20, 30] } }, (f) => {
      expect(runCol(f, col('a').add(col('b')))).toEqual([11, 22, 33]);
      expect(runCol(f, col('b').sub(col('a')))).toEqual([9, 18, 27]);
      expect(runCol(f, col('a').mul(col('b')))).toEqual([10, 40, 90]);
    });
  });
  it('null propagates through arithmetic', () => {
    withFrame({ a: { dtype: 'f64', values: [1, null, 3] }, b: { dtype: 'f64', values: [10, 20, null] } }, (f) => {
      expect(runCol(f, col('a').add(col('b')))).toEqual([11, null, null]);
      expect(runCol(f, col('a').add(1))).toEqual([2, null, 4]);
    });
  });
  it('NaN is a value, not a null (dtypes.md §4)', () => {
    withFrame({ a: { dtype: 'f64', values: [1, NaN, 3] } }, (f) => {
      expect(runCol(f, col('a').add(1))).toEqual([2, NaN, 4]);
    });
  });
  it('integer div/mod by zero → null; float div by zero → inf/nan (§3.2)', () => {
    withFrame({ c: { dtype: 'i32', values: [10, 20, 7] }, d: { dtype: 'i32', values: [2, 0, 2] } }, (f) => {
      expect(runCol(f, col('c').div(col('d')))).toEqual([5, null, 3]);
      expect(runCol(f, col('c').mod(col('d')))).toEqual([0, null, 1]);
      expect(runCol(f, col('c').div(0))).toEqual([null, null, null]);
    });
    withFrame({ a: { dtype: 'f64', values: [1, 0] }, b: { dtype: 'f64', values: [0, 0] } }, (f) => {
      expect(runCol(f, col('a').div(col('b')))).toEqual([Infinity, NaN]);
    });
  });
  it('int→float widening in mixed arithmetic', () => {
    withFrame({ c: { dtype: 'i32', values: [1, 2, 3] }, a: { dtype: 'f64', values: [0.5, 0.5, 0.5] } }, (f) => {
      expect(runCol(f, col('c').add(col('a')))).toEqual([1.5, 2.5, 3.5]);
    });
  });
  it('scalar on the left (non-commutative → materialised)', () => {
    withFrame({ a: { dtype: 'f64', values: [1, 2, 4] } }, (f) => {
      expect(runCol(f, lit(10).sub(col('a')))).toEqual([9, 8, 6]);
      expect(runCol(f, lit(12).div(col('a')))).toEqual([12, 6, 3]);
    });
  });
  it('neg preserves dtype', () => {
    withFrame({ c: { dtype: 'i32', values: [1, -2, 3] } }, (f) => {
      expect(runCol(f, col('c').neg())).toEqual([-1, 2, -3]);
    });
  });
});

describe('comparisons → bool column', () => {
  it('scalar comparison with null rows', () => {
    withFrame({ a: { dtype: 'f64', values: [1, 2, 3, null] } }, (f) => {
      expect(runCol(f, col('a').gt(2))).toEqual([false, false, true, null]);
      expect(runCol(f, col('a').le(2))).toEqual([true, true, false, null]);
    });
  });
  it('vector comparison propagates nulls from both sides', () => {
    withFrame({ a: { dtype: 'i32', values: [1, 5, null] }, b: { dtype: 'i32', values: [2, 3, 3] } }, (f) => {
      expect(runCol(f, col('a').gt(col('b')))).toEqual([false, true, null]);
    });
  });
  it('scalar on the left mirrors the operator', () => {
    withFrame({ a: { dtype: 'f64', values: [1, 2, 3] } }, (f) => {
      expect(runCol(f, lit(2).gt(col('a')))).toEqual([true, false, false]);
    });
  });
});

describe('boolean Kleene (dtypes.md §4.2)', () => {
  it('and/or/not follow the three-valued truth tables', () => {
    // a: T,T,F,F,N ; b: T,F,N,N,N  (encode via numeric predicates)
    withFrame(
      { p: { dtype: 'bool', values: [true, true, false, false, null] },
        q: { dtype: 'bool', values: [true, false, null, false, null] } },
      (f) => {
        expect(runCol(f, col('p').and(col('q')))).toEqual([true, false, false, false, null]);
        expect(runCol(f, col('p').or(col('q')))).toEqual([true, true, null, false, null]);
        expect(runCol(f, col('p').not())).toEqual([false, false, true, true, null]);
      },
    );
  });
  it('composes comparisons', () => {
    withFrame({ a: { dtype: 'f64', values: [1, 6, 3] }, b: { dtype: 'f64', values: [9, 1, 9] } }, (f) => {
      expect(runCol(f, col('a').gt(2).and(col('b').gt(2)))).toEqual([false, false, true]);
    });
  });
});

describe('null utilities', () => {
  it('isNull / notNull', () => {
    withFrame({ a: { dtype: 'f64', values: [1, null, NaN] } }, (f) => {
      expect(runCol(f, col('a').isNull())).toEqual([false, true, false]); // NaN is not null
      expect(runCol(f, col('a').notNull())).toEqual([true, false, true]);
    });
  });
  it('fillNull numeric / bool / utf8', () => {
    withFrame({ a: { dtype: 'f64', values: [1, null, 3] } }, (f) => {
      expect(runCol(f, col('a').fillNull(0))).toEqual([1, 0, 3]);
    });
    withFrame({ k: { dtype: 'bool', values: [true, null, false] } }, (f) => {
      expect(runCol(f, col('k').fillNull(true))).toEqual([true, true, false]);
    });
    withFrame({ s: { dtype: 'utf8', values: ['x', null, 'y'] } }, (f) => {
      expect(runCol(f, col('s').fillNull('z'))).toEqual(['x', 'z', 'y']);
      expect(runCol(f, col('s').fillNull('x'))).toEqual(['x', 'x', 'y']); // existing slot
    });
  });
});

describe('cast (dtypes.md §2)', () => {
  it('float→int truncates toward zero; out-of-range / NaN → null', () => {
    withFrame({ a: { dtype: 'f64', values: [2.9, -2.9, NaN, 3e9] } }, (f) => {
      expect(runCol(f, col('a').cast('i32'))).toEqual([2, -2, null, null]);
    });
  });
  it('int→float widening keeps value', () => {
    withFrame({ c: { dtype: 'i32', values: [1, 2, 3] } }, (f) => {
      expect(runCol(f, col('c').cast('f64'))).toEqual([1, 2, 3]);
    });
  });
  it('float→bool: x≠0→true, 0→false, NaN→null', () => {
    withFrame({ a: { dtype: 'f64', values: [0, 1.5, NaN] } }, (f) => {
      expect(runCol(f, col('a').cast('bool'))).toEqual([false, true, null]);
    });
  });
  it('propagates input nulls through a lossless cast', () => {
    withFrame({ c: { dtype: 'i32', values: [1, null, 3] } }, (f) => {
      expect(runCol(f, col('c').cast('f64'))).toEqual([1, null, 3]);
    });
  });
});

describe('string equality via dictionary index', () => {
  it('eq / ne against a present literal', () => {
    withFrame({ s: { dtype: 'utf8', values: ['a', 'b', 'a', null] } }, (f) => {
      expect(runCol(f, col('s').eq('a'))).toEqual([true, false, true, null]);
      expect(runCol(f, col('s').ne('a'))).toEqual([false, true, false, null]);
    });
  });
  it('eq against a missing literal → all-false; ne → all-true', () => {
    withFrame({ s: { dtype: 'utf8', values: ['a', 'b'] } }, (f) => {
      expect(runCol(f, col('s').eq('zzz'))).toEqual([false, false]);
      expect(runCol(f, col('s').ne('zzz'))).toEqual([true, true]);
    });
  });
});

describe('aggregations (dtypes.md §4.3)', () => {
  it('numeric reductions skip nulls', () => {
    withFrame({ a: { dtype: 'f64', values: [1, null, 3, 5] } }, (f) => {
      expect(runScalar(f, col('a').sum())).toBe(9);
      expect(runScalar(f, col('a').mean())).toBe(3);
      expect(runScalar(f, col('a').min())).toBe(1);
      expect(runScalar(f, col('a').max())).toBe(5);
      expect(runScalar(f, col('a').count())).toBe(3);
    });
  });
  it('empty / all-null edges', () => {
    withFrame({ a: { dtype: 'f64', values: [null, null] } }, (f) => {
      expect(runScalar(f, col('a').sum())).toBe(0); // additive identity
      expect(runScalar(f, col('a').mean())).toBe(null);
      expect(runScalar(f, col('a').min())).toBe(null);
      expect(runScalar(f, col('a').count())).toBe(0);
      expect(runScalar(f, col('a').std())).toBe(null); // < 2 non-null
    });
  });
  it('std / var, ddof=1', () => {
    withFrame({ a: { dtype: 'f64', values: [2, 4, 4, 4, 5, 5, 7, 9] } }, (f) => {
      expect(runScalar(f, col('a').var())).toBeCloseTo(4.571428, 4);
      expect(runScalar(f, col('a').std())).toBeCloseTo(2.13809, 4);
    });
  });
  it('nunique counts distinct non-null; NaN counts once', () => {
    withFrame({ a: { dtype: 'f64', values: [1, 1, 2, null, NaN, NaN] } }, (f) => {
      expect(runScalar(f, col('a').nunique())).toBe(3);
    });
  });
  it('first / last skip nulls', () => {
    withFrame({ a: { dtype: 'f64', values: [null, 7, 8, null] } }, (f) => {
      expect(runScalar(f, col('a').first())).toBe(7);
      expect(runScalar(f, col('a').last())).toBe(8);
    });
  });
  it('utf8 count / nunique / first / last', () => {
    withFrame({ s: { dtype: 'utf8', values: ['a', 'b', 'a', null, 'c'] } }, (f) => {
      expect(runScalar(f, col('s').count())).toBe(4);
      expect(runScalar(f, col('s').nunique())).toBe(3);
      expect(runScalar(f, col('s').first())).toBe('a');
      expect(runScalar(f, col('s').last())).toBe('c');
    });
  });
  it('sum of i32 widens to f64', () => {
    withFrame({ c: { dtype: 'i32', values: [1, 2, 3] } }, (f) => {
      const p = compile(col('c').sum(), f).execute();
      expect(p.scalar!.value).toBe(6);
      expect(p.scalar!.dtype).toBe('f64');
    });
  });
  it('aggregate broadcast into an elementwise expression', () => {
    withFrame({ a: { dtype: 'f64', values: [1, 2, 3] } }, (f) => {
      // a - mean(a) = a - 2
      expect(runCol(f, col('a').sub(col('a').mean()))).toEqual([-1, 0, 1]);
    });
  });
});

describe('filter (compare → filter fusion)', () => {
  it('keeps predicate-true rows, drops false and null', () => {
    withFrame(
      { a: { dtype: 'f64', values: [1, 5, 2, 8, null] }, s: { dtype: 'utf8', values: ['p', 'q', 'r', 's', 't'] } },
      (f) => {
        const sel = compileFilter(col('a').gt(3), f).execute();
        expect(sel.count).toBe(2);
        const a = takeColumn(f.ctx, sel.compact(f.getColumn('a')!));
        const s = takeColumn(f.ctx, sel.compact(f.getColumn('s')!));
        sel.free();
        expect(a).toEqual([5, 8]);
        expect(s).toEqual(['q', 's']);
      },
    );
  });
  it('compacts null-bearing columns', () => {
    withFrame(
      { a: { dtype: 'i32', values: [1, 2, 3, 4] }, b: { dtype: 'i32', values: [null, 20, null, 40] } },
      (f) => {
        const sel = compileFilter(col('a').ge(2), f).execute();
        expect(sel.count).toBe(3);
        const b = takeColumn(f.ctx, sel.compact(f.getColumn('b')!));
        sel.free();
        expect(b).toEqual([20, null, 40]);
      },
    );
  });
  it('filters on a boolean combination', () => {
    withFrame({ a: { dtype: 'f64', values: [1, 5, 2, 8] }, b: { dtype: 'f64', values: [9, 9, 1, 9] } }, (f) => {
      const sel = compileFilter(col('a').gt(3).and(col('b').gt(3)), f).execute();
      const a = takeColumn(f.ctx, sel.compact(f.getColumn('a')!));
      sel.free();
      expect(a).toEqual([5, 8]);
    });
  });
});

describe('empty frame', () => {
  it('handles zero rows', () => {
    withFrame({ a: { dtype: 'f64', values: [] } }, (f) => {
      expect(runCol(f, col('a').add(1))).toEqual([]);
      expect(runScalar(f, col('a').sum())).toBe(0);
      expect(runScalar(f, col('a').mean())).toBe(null);
    });
  });
});
