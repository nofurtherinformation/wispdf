/**
 * Fusion verification by plan inspection (P3.1 §4). Asserts on the ordered kernel
 * list + allocation counts, not just output equality:
 *   (a) compare → filter: one mask + one compaction per column, no bool column.
 *   (b) elementwise chains reuse ONE data buffer instead of one per node.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { col } from '../../src/expr/ast.js';
import { compile, compileFilter } from '../../src/expr/compile.js';
import { takeColumn } from './helper.js';
import { loadEnv, TestFrame, type TestEnv, type ColSpec } from './helper.js';

let env: TestEnv;
beforeAll(async () => { env = await loadEnv(); });

function frame(specs: Record<string, ColSpec>): TestFrame {
  return new TestFrame(env, specs);
}

describe('(b) elementwise chains reuse one output buffer', () => {
  it('a 3-op scalar chain allocates exactly one data buffer', () => {
    const f = frame({ a: { dtype: 'f64', values: [1, 2, 3, 4] } });
    try {
      const p = compile(col('a').add(1).mul(2).sub(3), f).execute();
      expect(p.stats.dataAllocations).toBe(1);
      expect(p.stats.kernels).toEqual(['add_f64_scalar', 'mul_f64_scalar', 'sub_f64_scalar']);
      takeColumn(f.ctx, p.column!);
    } finally {
      f.free();
    }
  });

  it('a vector chain reuses the first temp for later nodes', () => {
    const f = frame({
      a: { dtype: 'f64', values: [1, 2, 3] },
      b: { dtype: 'f64', values: [4, 5, 6] },
      c: { dtype: 'f64', values: [7, 8, 9] },
    });
    try {
      const p = compile(col('a').add(col('b')).mul(col('c')), f).execute();
      expect(p.stats.dataAllocations).toBe(1); // add allocates; mul writes in place
      expect(p.stats.kernels).toEqual(['add_f64', 'mul_f64']);
      takeColumn(f.ctx, p.column!);
    } finally {
      f.free();
    }
  });

  it('a longer chain still allocates exactly one buffer', () => {
    const f = frame({ a: { dtype: 'i32', values: [10, 20, 30] } });
    try {
      const p = compile(col('a').add(1).add(1).add(1).add(1), f).execute();
      expect(p.stats.dataAllocations).toBe(1);
      expect(p.stats.kernelCalls).toBe(4);
      takeColumn(f.ctx, p.column!);
    } finally {
      f.free();
    }
  });
});

describe('(a) compare → filter fuses to one mask + one compaction per column', () => {
  it('a bare comparison emits a single mask; no bool column is materialised', () => {
    const f = frame({
      a: { dtype: 'f64', values: [1, 5, 2, 8] },
      b: { dtype: 'f64', values: [10, 20, 30, 40] },
    });
    try {
      const sel = compileFilter(col('a').gt(3), f).execute();
      // predicate lowered to exactly one comparison mask, zero data (bool-column) allocs
      expect(sel.stats.kernels).toEqual(['gt_f64_scalar_mask']);
      expect(sel.stats.maskAllocations).toBe(1);
      expect(sel.stats.dataAllocations).toBe(0);
      expect(sel.stats.kernels).not.toContain('expand_mask_bool');

      // one compaction (filter kernel) per column
      const a = takeColumn(f.ctx, sel.compact(f.getColumn('a')!));
      const b = takeColumn(f.ctx, sel.compact(f.getColumn('b')!));
      expect(sel.stats.kernels).toEqual(['gt_f64_scalar_mask', 'filter_f64', 'filter_f64']);
      expect(sel.stats.kernels).not.toContain('expand_mask_bool');
      sel.free();

      expect(a).toEqual([5, 8]);
      expect(b).toEqual([20, 40]);
    } finally {
      f.free();
    }
  });

  it('vector comparison predicate is also a single mask', () => {
    const f = frame({
      a: { dtype: 'i32', values: [1, 9, 3] },
      b: { dtype: 'i32', values: [2, 2, 5] },
    });
    try {
      const sel = compileFilter(col('a').gt(col('b')), f).execute();
      expect(sel.stats.kernels).toEqual(['gt_i32_mask']);
      expect(sel.stats.dataAllocations).toBe(0);
      const a = takeColumn(f.ctx, sel.compact(f.getColumn('a')!));
      sel.free();
      expect(a).toEqual([9]);
    } finally {
      f.free();
    }
  });
});
