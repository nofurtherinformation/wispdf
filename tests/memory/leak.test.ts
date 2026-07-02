import { describe, it, expect } from 'vitest';
import { loadForTest } from './helper.js';

// Freelist reuse must keep the heap high-water bounded: many alloc/free cycles
// of varied sizes must not grow linear memory monotonically (Phase-1 gate).
describe('allocator leak / freelist reuse', () => {
  it('reuses freed blocks and does not grow memory monotonically', async () => {
    const mod = await loadForTest();
    const pages = () => mod.memory.buffer.byteLength / 65536;

    // A fixed but varied set of sizes exercised every cycle. Each cycle frees
    // everything it allocates, so with coalescing + top-trim the high-water is
    // reached once and then held.
    const sizes = [24, 100, 333, 1024, 4096, 40, 777, 8000];

    const cycle = (c: number): void => {
      const s1 = sizes[c % sizes.length]!;
      const s2 = sizes[(c + 3) % sizes.length]!;
      const a = mod.alloc(s1);
      const b = mod.alloc(s2); // pins `a` so free(a) exercises the free list
      mod.free(a);
      const c2 = mod.alloc(s1);
      expect(c2).toBe(a); // proves freelist reuse (not a fresh bump)
      mod.free(c2);
      mod.free(b);
    };

    // Warm up to the steady-state high-water.
    for (let c = 0; c < sizes.length * 4; c++) cycle(c);
    const checkpoint = pages();

    // Sustained churn must not grow memory further.
    for (let c = 0; c < 5000; c++) cycle(c);

    expect(pages()).toBe(checkpoint);
  });

  it('drains fully when all live blocks are freed (batch cycles)', async () => {
    const mod = await loadForTest();
    const pages = () => mod.memory.buffer.byteLength / 65536;
    const batch = [16, 4096, 240, 64, 1500, 32, 900, 12000];

    const runBatch = (): void => {
      const ps = batch.map((s) => mod.alloc(s));
      for (const p of ps) expect(p).not.toBe(0);
      // Free high-to-low so top-trim reclaims the whole batch each cycle.
      for (let i = ps.length - 1; i >= 0; i--) mod.free(ps[i]!);
    };

    for (let c = 0; c < 20; c++) runBatch();
    const checkpoint = pages();
    for (let c = 0; c < 2000; c++) runBatch();
    expect(pages()).toBe(checkpoint);
  });
});
