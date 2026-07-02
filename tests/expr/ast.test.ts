/**
 * Unit tests for the AST: immutability, chainability, and rendering (P3.1 §1).
 */
import { describe, it, expect } from 'vitest';
import { col, lit, Expr } from '../../src/expr/ast.js';

describe('AST construction', () => {
  it('col / lit build the expected nodes', () => {
    expect(col('a').node).toEqual({ kind: 'col', name: 'a' });
    expect(lit(5).node).toEqual({ kind: 'lit', value: 5, dtype: null });
    expect(lit(5, 'f32').node).toEqual({ kind: 'lit', value: 5, dtype: 'f32' });
    expect(lit('x').node).toEqual({ kind: 'lit', value: 'x', dtype: null });
  });

  it('wraps raw scalars as literals in operators', () => {
    const e = col('a').gt(5);
    expect(e.node.kind).toBe('compare');
    if (e.node.kind === 'compare') {
      expect(e.node.op).toBe('gt');
      expect(e.node.right.node).toEqual({ kind: 'lit', value: 5, dtype: null });
    }
  });

  it('builds the full spec §4 surface', () => {
    const a = col('a');
    for (const e of [
      a.add(1), a.sub(1), a.mul(2), a.div(2), a.mod(2), a.neg(),
      a.gt(1), a.ge(1), a.lt(1), a.le(1), a.eq(1), a.ne(1),
      a.gt(1).and(a.lt(2)), a.gt(1).or(a.lt(2)), a.gt(1).not(),
      a.isNull(), a.notNull(), a.fillNull(0), a.cast('f32'),
      a.sum(), a.mean(), a.min(), a.max(), a.count(), a.nunique(),
      a.std(), a.var(), a.first(), a.last(),
    ]) {
      expect(e).toBeInstanceOf(Expr);
    }
  });
});

describe('immutability', () => {
  it('nodes and expressions are frozen', () => {
    const e = col('a').add(1);
    expect(Object.isFrozen(e)).toBe(true);
    expect(Object.isFrozen(e.node)).toBe(true);
  });

  it('chaining returns a new expression, never mutating the receiver', () => {
    const base = col('a');
    const derived = base.add(1);
    expect(derived).not.toBe(base);
    expect(base.node).toEqual({ kind: 'col', name: 'a' });
  });
});

describe('toString', () => {
  it('renders a readable, unambiguous form', () => {
    expect(col('a').toString()).toBe('col("a")');
    expect(col('a').add(5).toString()).toBe('(col("a") + lit(5))');
    expect(col('a').gt(5).and(col('b').eq('x')).toString()).toBe(
      'col("a").gt(lit(5)).and(col("b").eq(lit("x")))',
    );
    expect(col('a').neg().toString()).toBe('(-col("a"))');
    expect(col('a').cast('f32').sum().toString()).toBe('col("a").cast(f32).sum()');
  });
});
