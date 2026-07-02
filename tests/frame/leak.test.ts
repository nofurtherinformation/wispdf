/**
 * Frame-layer buffer-lifecycle checks: after a materialising op's result is `dispose()`d,
 * the arena returns to baseline (freelist probe pointer stable, no page growth). Also
 * verifies the reference-counted share model: a child frame keeps buffers alive after the
 * parent is disposed.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { loadRuntimeForTest, makeDF } from './helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import { col } from '../../src/expr/index.js';

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

const probe = (r: DfRuntime): number => {
  const p = r.ctx.mod.alloc(48);
  r.ctx.mod.free(p);
  return p;
};
const pages = (r: DfRuntime): number => r.ctx.mod.memory.buffer.byteLength / 65536;

describe('no leaks: materialising ops return to baseline after dispose', () => {
  it('filter / withColumn / sortValues / groupby / join / filterFn', () => {
    const base = makeDF(
      rt,
      { a: [1, 5, 2, 8, 3, 9], g: ['x', 'y', 'x', 'y', 'x', 'y'], v: [1, 2, 3, 4, 5, 6] },
      { a: 'f64', g: 'utf8', v: 'f64' },
    );
    const other = makeDF(rt, { g: ['x', 'y'], w: [10, 20] }, { g: 'utf8', w: 'f64' });

    const cycle = (): void => {
      base.filter(col('a').gt(3)).dispose();
      base.withColumn('a2', col('a').mul(2)).dispose();
      base.sortValues('a', { descending: true }).dispose();
      base.groupby('g').agg({ a: 'sum', v: ['mean', 'max'] }).dispose();
      base.join(other, { on: 'g', how: 'left' }).dispose();
      base.filterFn((row) => (row.a as number) > 3).dispose();
      base.select(['a', 'g']).dispose();
      base.head(3).dispose();
    };

    try {
      for (let i = 0; i < 3; i++) cycle(); // warm up freelist high-water
      const before = probe(rt);
      const p0 = pages(rt);
      for (let i = 0; i < 40; i++) cycle();
      expect(pages(rt)).toBe(p0);
      expect(probe(rt)).toBe(before);
    } finally {
      base.dispose();
      other.dispose();
    }
  });

  it('a shared child outlives its disposed parent (refcount)', () => {
    const parent = makeDF(rt, { a: [1, 2, 3], b: [4, 5, 6] }, { a: 'f64', b: 'f64' });
    const child = parent.select(['a']);
    parent.dispose(); // buffers must NOT be freed yet — child still holds a ref
    expect(child.toColumns().a).toEqual([1, 2, 3]);
    child.dispose();
  });

  it('double dispose is a no-op', () => {
    const df = makeDF(rt, { a: [1] }, { a: 'f64' });
    df.dispose();
    expect(() => df.dispose()).not.toThrow();
  });
});
