import { describe, it, expect } from 'vitest';
import { ctxForTest } from './helper.js';
import {
  createColumn,
  columnToArray,
  sliceColumn,
  freeColumn,
} from '../../src/memory/index.js';

// ADR-001 / ABI §2: a `memory.grow` (triggered here by a large alloc) detaches
// every view. The column layer reads bytes only through `viewOf`, so a column
// created BEFORE a grow must still read back correctly AFTER it — for numeric
// data, the validity bitmap, and a dictionary-encoded utf8 column + its slice.
describe('columns survive a memory.grow mid-operation', () => {
  it('re-reads numeric + validity + dictionary buffers after a forced grow', async () => {
    const ctx = await ctxForTest(false);

    const nums: (number | null)[] = [0.5, null, 2.5, NaN, -3.5, null, 6.5];
    const strs: (string | null)[] = ['x', 'y', null, 'x', 'z', 'y', 'x'];
    const numCol = createColumn(ctx, 'f64', nums);
    const strCol = createColumn(ctx, 'utf8', strs);
    const slice = sliceColumn(strCol, 2, 6);

    const gen0 = ctx.mod.mem_generation();
    const bufBefore = ctx.mod.memory.buffer;

    // Force a grow: allocate far more than currently mapped.
    const big = ctx.mod.alloc(96 * 1024 * 1024);
    expect(big).not.toBe(0);
    expect(ctx.mod.mem_generation()).toBeGreaterThan(gen0);
    expect(ctx.mod.memory.buffer).not.toBe(bufBefore); // buffer really swapped

    // All reads rebuild views over the new buffer and see preserved bytes.
    expect(columnToArray(ctx, numCol)).toEqual(nums);
    expect(columnToArray(ctx, strCol)).toEqual(strs);
    expect(columnToArray(ctx, slice)).toEqual(strs.slice(2, 6));

    ctx.mod.free(big);
    freeColumn(ctx, numCol);
    freeColumn(ctx, strCol);
  });
});
