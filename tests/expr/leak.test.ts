/**
 * Buffer-lifecycle leak checks (P3.1 §3). Uses the Phase-1 arena's freelist reuse as
 * the "arena stat": after a balanced plan the next allocation returns the *same*
 * pointer, and the heap high-water does not grow. A leaked temp would push the top or
 * occupy the probe slot.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { col, lit, type Expr } from '../../src/expr/ast.js';
import { compile, compileFilter } from '../../src/expr/compile.js';
import { freeColumn } from '../../src/memory/column.js';
import { loadEnv, TestFrame, type TestEnv } from './helper.js';

let env: TestEnv;
beforeAll(async () => { env = await loadEnv(); });

/** Alloc+free a fixed block; its pointer is a fingerprint of the freelist state. */
function probe(f: TestFrame): number {
  const p = f.ctx.mod.alloc(48);
  f.ctx.mod.free(p);
  return p;
}
const pages = (f: TestFrame): number => f.ctx.mod.memory.buffer.byteLength / 65536;

function runAndFree(f: TestFrame, expr: Expr): void {
  const p = compile(expr, f).execute();
  if (p.kind === 'column') freeColumn(f.ctx, p.column!);
}

describe('no leaks: the arena returns to baseline after each plan', () => {
  const EXPRS: Expr[] = [
    col('a').add(1).mul(2).sub(3),
    col('a').add(col('b')),
    col('c').div(col('c')),
    col('a').gt(1).and(col('b').lt(5)),
    col('a').gt(1).or(col('b').lt(5)).not(),
    col('a').isNull(),
    col('a').fillNull(0),
    col('a').cast('i32'),
    col('c').cast('f64').add(0.5),
    col('s').eq('x'),
    col('s').fillNull('zz'), // new dictionary slot → dict extension temp
    col('a').sum(),
    col('a').mean(),
    col('s').nunique(),
    col('a').sub(col('a').mean()),
    lit(10).sub(col('a')),
  ];

  it('every expression leaves the freelist pointer stable', () => {
    const f = new TestFrame(env, {
      a: { dtype: 'f64', values: [1, null, 3, 5] },
      b: { dtype: 'f64', values: [2, 4, 6, 8] },
      c: { dtype: 'i32', values: [3, 0, 4, 2] },
      s: { dtype: 'utf8', values: ['x', 'y', null, 'x'] },
    });
    try {
      runAndFree(f, col('a').add(1)); // warm up the high-water
      for (const expr of EXPRS) {
        const before = probe(f);
        runAndFree(f, expr);
        const after = probe(f);
        expect(after, `leak after ${expr}`).toBe(before);
      }
    } finally {
      f.free();
    }
  });

  it('filter (mask + compaction) leaves no leak across many iterations', () => {
    const f = new TestFrame(env, {
      a: { dtype: 'f64', values: [1, 5, 2, 8, 3, 9] },
      b: { dtype: 'i32', values: [null, 20, null, 40, 50, null] },
    });
    try {
      // warm up
      for (let i = 0; i < 5; i++) {
        const sel = compileFilter(col('a').gt(3), f).execute();
        freeColumn(f.ctx, sel.compact(f.getColumn('a')!));
        freeColumn(f.ctx, sel.compact(f.getColumn('b')!));
        sel.free();
      }
      const before = probe(f);
      const p0 = pages(f);
      for (let i = 0; i < 500; i++) {
        const sel = compileFilter(col('a').gt(3), f).execute();
        freeColumn(f.ctx, sel.compact(f.getColumn('a')!));
        freeColumn(f.ctx, sel.compact(f.getColumn('b')!));
        sel.free();
      }
      expect(pages(f)).toBe(p0); // no monotonic growth
      expect(probe(f)).toBe(before); // freelist unchanged
    } finally {
      f.free();
    }
  });

  it('compiled column results are independent (freeing one does not corrupt sources)', () => {
    const f = new TestFrame(env, { a: { dtype: 'f64', values: [1, 2, 3] } });
    try {
      const p = compile(col('a').add(10), f).execute();
      freeColumn(f.ctx, p.column!);
      // source 'a' is untouched
      const again = compile(col('a').mul(2), f).execute();
      expect(again.kind).toBe('column');
      freeColumn(f.ctx, again.column!);
    } finally {
      f.free();
    }
  });
});
