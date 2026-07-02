/**
 * Unit tests for type & validity resolution (dtypes.md §3.1/§5) and error messages
 * (spec §4 ergonomics). Pure — no wasm.
 */
import { describe, it, expect } from 'vitest';
import { col, lit } from '../../src/expr/ast.js';
import { inferType, resolve, schemaOf, aggResult } from '../../src/expr/dtypes.js';
import { ExprError } from '../../src/expr/errors.js';
import type { DType } from '../../src/memory/dtype.js';

const S = schemaOf({
  a: 'f64', b: 'f64', c: 'i32', d: 'i32', e: 'f32', g: 'u32', s: 'utf8', k: 'bool',
});

const typeOf = (e: Parameters<typeof inferType>[0]): DType => inferType(e, S);

describe('arithmetic result dtype (dtypes.md §3.1)', () => {
  it('same-dtype keeps dtype', () => {
    expect(typeOf(col('a').add(col('b')))).toBe('f64');
    expect(typeOf(col('c').mul(col('d')))).toBe('i32');
    expect(typeOf(col('g').sub(col('g')))).toBe('u32');
  });

  it('int→float widening is the only implicit conversion', () => {
    expect(typeOf(col('c').add(col('a')))).toBe('f64'); // i32 ⊕ f64
    expect(typeOf(col('g').mul(col('a')))).toBe('f64'); // u32 ⊕ f64
    expect(typeOf(col('c').add(col('e')))).toBe('f32'); // i32 ⊕ f32
  });

  it('unsupported mixes name both dtypes and the op', () => {
    expect(() => typeOf(col('c').add(col('g')))).toThrowError(/i32.*u32|u32.*i32/);
    expect(() => typeOf(col('e').add(col('a')))).toThrowError(/f32.*f64|f64.*f32/);
    let msg = '';
    try { typeOf(col('c').add(col('g'))); } catch (e) { msg = (e as Error).message; }
    expect(msg).toContain('add');
    expect(msg).toContain('i32');
    expect(msg).toContain('u32');
  });

  it('arithmetic on bool / utf8 is rejected', () => {
    expect(() => typeOf(col('k').add(lit(1)))).toThrowError(ExprError);
    expect(() => typeOf(col('s').add(col('s')))).toThrowError(/utf8/);
  });

  it('div/mod on two integers stay integer (dtypes.md §3.2)', () => {
    expect(typeOf(col('c').div(col('d')))).toBe('i32');
    expect(typeOf(col('c').mod(col('d')))).toBe('i32');
  });
});

describe('literal typing', () => {
  it('an integer literal adopts an integer column dtype', () => {
    expect(typeOf(col('c').add(5))).toBe('i32');
    expect(typeOf(col('c').gt(5))).toBe('bool');
  });
  it('a fractional literal widens an integer column to f64', () => {
    expect(typeOf(col('c').add(5.5))).toBe('f64');
    expect(typeOf(col('c').gt(5.5))).toBe('bool'); // compares as f64 under the hood
  });
  it('a literal adopts a float column dtype', () => {
    expect(typeOf(col('a').add(2))).toBe('f64');
    expect(typeOf(col('e').mul(2))).toBe('f32');
  });
});

describe('comparisons require exact dtype (no widening)', () => {
  it('two columns of different dtype error', () => {
    expect(() => typeOf(col('c').gt(col('a')))).toThrowError(/identical dtypes|cast/);
  });
  it('same-dtype numeric comparisons yield bool', () => {
    expect(typeOf(col('a').lt(col('b')))).toBe('bool');
  });
  it('utf8 supports only eq/ne against a string literal', () => {
    expect(typeOf(col('s').eq('x'))).toBe('bool');
    expect(typeOf(col('s').ne('x'))).toBe('bool');
    expect(() => typeOf(col('s').gt('x'))).toThrowError(/string ordering|utf8/);
    expect(() => typeOf(col('s').eq(col('s')))).toThrowError(/column-vs-column|unification|not supported/);
  });
});

describe('boolean ops', () => {
  it('and/or require boolean operands', () => {
    expect(typeOf(col('a').gt(1).and(col('b').lt(2)))).toBe('bool');
    expect(() => typeOf(col('a').and(col('b')))).toThrowError(/boolean/);
  });
  it('not requires a boolean operand', () => {
    expect(typeOf(col('a').gt(1).not())).toBe('bool');
    expect(() => typeOf(col('a').not())).toThrowError(/boolean/);
  });
});

describe('cast matrix (dtypes.md §2)', () => {
  it('allowed numeric/bool casts resolve to the target', () => {
    expect(typeOf(col('a').cast('i32'))).toBe('i32');
    expect(typeOf(col('c').cast('f64'))).toBe('f64');
    expect(typeOf(col('k').cast('i32'))).toBe('i32');
    expect(typeOf(col('a').cast('f64'))).toBe('f64'); // identity kept in the IR
  });
  it('numeric↔utf8 casts throw a helpful error', () => {
    expect(() => typeOf(col('a').cast('utf8'))).toThrowError(/utf8/);
    expect(() => typeOf(col('s').cast('f64'))).toThrowError(/utf8/);
  });
});

describe('fillNull value must match dtype (dtypes.md §4.5)', () => {
  it('accepts a matching value', () => {
    expect(typeOf(col('c').fillNull(0))).toBe('i32');
    expect(typeOf(col('a').fillNull(NaN))).toBe('f64');
    expect(typeOf(col('s').fillNull('x'))).toBe('utf8');
    expect(typeOf(col('k').fillNull(false))).toBe('bool');
  });
  it('rejects a mismatched value', () => {
    expect(() => typeOf(col('c').fillNull('x'))).toThrowError(ExprError);
    expect(() => typeOf(col('c').fillNull(2.5))).toThrowError(/i32/);
    expect(() => typeOf(col('s').fillNull(1))).toThrowError(/utf8/);
  });
});

describe('aggregation result dtypes (dtypes.md §4.3)', () => {
  it('follows the matrix', () => {
    expect(aggResult('sum', 'i32')).toBe('f64');
    expect(aggResult('sum', 'f32')).toBe('f32');
    expect(aggResult('mean', 'i32')).toBe('f64');
    expect(aggResult('min', 'i32')).toBe('i32');
    expect(aggResult('count', 'utf8')).toBe('i32');
    expect(aggResult('nunique', 'utf8')).toBe('i32');
    expect(aggResult('first', 'utf8')).toBe('utf8');
  });
  it('rejects unsupported aggregation/dtype pairs', () => {
    expect(() => aggResult('sum', 'utf8')).toThrowError(/utf8/);
    expect(() => aggResult('min', 'utf8')).toThrowError(/utf8/);
    expect(() => aggResult('mean', 'bool')).toThrowError(/bool/);
  });
  it('resolves through the expression surface', () => {
    expect(typeOf(col('c').sum())).toBe('f64');
    expect(typeOf(col('s').nunique())).toBe('i32');
    expect(typeOf(col('a').std())).toBe('f64');
  });
});

describe('unknown column errors suggest the nearest match', () => {
  it('suggests a close name', () => {
    let msg = '';
    try { typeOf(col('aa')); } catch (e) { msg = (e as Error).message; }
    expect(msg).toContain("unknown column 'aa'");
    expect(msg).toContain("Did you mean 'a'");
  });
  it('lists known columns', () => {
    let msg = '';
    try { typeOf(col('zzzzz')); } catch (e) { msg = (e as Error).message; }
    expect(msg).toContain('Known columns');
  });
});

describe('cast insertion in the typed IR', () => {
  it('inserts an explicit cast for int→float widening', () => {
    const t = resolve(col('c').add(col('a')), S);
    expect(t.kind).toBe('arith');
    if (t.kind === 'arith') {
      expect(t.dtype).toBe('f64');
      // the i32 operand is wrapped in a cast to f64; the f64 operand is not
      expect(t.left.kind).toBe('cast');
      expect(t.right.kind).toBe('col');
    }
  });
  it('does not wrap identity casts', () => {
    const t = resolve(col('a').add(col('b')), S);
    if (t.kind === 'arith') {
      expect(t.left.kind).toBe('col');
      expect(t.right.kind).toBe('col');
    }
  });
});
