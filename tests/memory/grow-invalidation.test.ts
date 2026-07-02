import { describe, it, expect } from 'vitest';
import { loadForTest } from './helper.js';
import { createViewOf, type ColumnBuffer } from '../../src/memory/views.js';

// ADR-001 / ABI §2: memory.grow detaches every view. viewOf() must detect the
// generation change and rebuild over the current buffer, reading correct data.
describe('grow invalidation (viewOf)', () => {
  it('detects the generation bump and rebuilds views with correct data', async () => {
    const mod = await loadForTest();
    const viewOf = createViewOf(mod);

    const N = 8;
    const ptr = mod.alloc(N * 8);
    const col: ColumnBuffer = { ptr, length: N, dtype: 'f64' };

    const v1 = viewOf(col) as Float64Array;
    for (let i = 0; i < N; i++) v1[i] = i + 0.5;
    // Same generation -> same cached view returned.
    expect(viewOf(col)).toBe(v1);

    const gen0 = mod.mem_generation();
    const bufBefore = mod.memory.buffer;

    // Force a grow: allocate far more than current memory.
    const big = mod.alloc(64 * 1024 * 1024);
    expect(big).not.toBe(0);

    const gen1 = mod.mem_generation();
    expect(gen1).toBeGreaterThan(gen0);
    expect(mod.memory.buffer).not.toBe(bufBefore);
    // The old view was detached by the grow.
    expect(v1.byteLength).toBe(0);

    // viewOf rebuilds over the new buffer and reads the preserved data.
    const v2 = viewOf(col) as Float64Array;
    // (compare identity via a boolean: v1 is detached and must not be handed to
    // the matcher, which would iterate it and throw.)
    expect(v2 === v1).toBe(false);
    expect(v2.buffer).toBe(mod.memory.buffer);
    expect(v2.byteLength).toBe(N * 8);
    for (let i = 0; i < N; i++) expect(v2[i]).toBe(i + 0.5);
    expect(viewOf.generation()).toBe(gen1);
  });

  it('rebuilds ALL registered views on a single generation change', async () => {
    const mod = await loadForTest();
    const viewOf = createViewOf(mod);

    const cols: ColumnBuffer[] = [];
    // Register several columns of different dtypes and seed them.
    const a: ColumnBuffer = { ptr: mod.alloc(4 * 8), length: 4, dtype: 'f64' };
    const b: ColumnBuffer = { ptr: mod.alloc(4 * 4), length: 4, dtype: 'i32' };
    const c: ColumnBuffer = { ptr: mod.alloc(4), length: 4, dtype: 'u8' };
    cols.push(a, b, c);

    const va = viewOf(a) as Float64Array;
    const vb = viewOf(b) as Int32Array;
    const vc = viewOf(c) as Uint8Array;
    for (let i = 0; i < 4; i++) {
      va[i] = i * 1.5;
      vb[i] = i * 100;
      vc[i] = i + 1;
    }

    const gen0 = mod.mem_generation();
    mod.alloc(64 * 1024 * 1024); // grow
    expect(mod.mem_generation()).toBeGreaterThan(gen0);

    const va2 = viewOf(a) as Float64Array;
    const vb2 = viewOf(b) as Int32Array;
    const vc2 = viewOf(c) as Uint8Array;
    // All rebuilt over the new buffer...
    expect(va2.buffer).toBe(mod.memory.buffer);
    expect(vb2.buffer).toBe(mod.memory.buffer);
    expect(vc2.buffer).toBe(mod.memory.buffer);
    // ...and none is the stale (detached) instance.
    expect(va2 === va).toBe(false);
    expect(vb2 === vb).toBe(false);
    expect(vc2 === vc).toBe(false);
    // ...with correct preserved data.
    for (let i = 0; i < 4; i++) {
      expect(va2[i]).toBe(i * 1.5);
      expect(vb2[i]).toBe(i * 100);
      expect(vc2[i]).toBe(i + 1);
    }
  });
});
