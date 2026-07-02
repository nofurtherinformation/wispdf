import { describe, it, expect, beforeEach } from 'vitest';
import { BUILDS, loadForTest } from './helper.js';
import type { WasmMemoryModule } from '../../src/memory/loader.js';

// The allocator (ABI §3/§9) is compiled into both builds identically; run the
// full suite against each. A fresh module per test isolates allocator state.
for (const { label, simd } of BUILDS) {
  describe(`arena allocator (${label})`, () => {
    let mod: WasmMemoryModule;
    beforeEach(async () => {
      mod = await loadForTest(simd);
    });

    it('loads the requested build and exports the ABI', () => {
      expect(mod.simd).toBe(simd);
      expect(mod.memory).toBeInstanceOf(WebAssembly.Memory);
      expect(typeof mod.alloc).toBe('function');
      expect(typeof mod.free).toBe('function');
      expect(typeof mod.realloc).toBe('function');
      expect(typeof mod.mem_generation).toBe('function');
    });

    it('returns 16-byte-aligned, non-null pointers', () => {
      for (const size of [1, 7, 8, 15, 16, 17, 100, 4096, 12345]) {
        const p = mod.alloc(size);
        expect(p, `alloc(${size})`).not.toBe(0);
        expect(p % 16, `alloc(${size}) alignment`).toBe(0);
      }
    });

    it('alloc(0) returns a valid aligned pointer', () => {
      const p = mod.alloc(0);
      expect(p).not.toBe(0);
      expect(p % 16).toBe(0);
    });

    it('free(0) is a no-op and leaves the allocator usable', () => {
      expect(() => mod.free(0)).not.toThrow();
      const p = mod.alloc(64);
      expect(p).not.toBe(0);
      expect(p % 16).toBe(0);
    });

    it('reuses a freed block (freelist first-fit)', () => {
      const a = mod.alloc(256);
      const b = mod.alloc(256); // pins `a` below the heap top
      expect(b).not.toBe(0);
      mod.free(a);
      const c = mod.alloc(256);
      expect(c).toBe(a); // served from the free list, not bumped
      mod.free(b);
      mod.free(c);
    });

    it('realloc(0, n) behaves like alloc(n)', () => {
      const p = mod.realloc(0, 128);
      expect(p).not.toBe(0);
      expect(p % 16).toBe(0);
    });

    it('realloc grows in place (top block) preserving contents', () => {
      const n = 64;
      const p = mod.alloc(n);
      const u = new Uint8Array(mod.memory.buffer, p, n);
      for (let i = 0; i < n; i++) u[i] = (i * 7 + 1) & 0xff;

      const p2 = mod.realloc(p, 4096);
      expect(p2).not.toBe(0);
      expect(p2 % 16).toBe(0);

      const u2 = new Uint8Array(mod.memory.buffer, p2, 4096);
      for (let i = 0; i < n; i++) {
        expect(u2[i], `byte ${i}`).toBe((i * 7 + 1) & 0xff);
      }
    });

    it('realloc grows with a move (block pinned) preserving contents', () => {
      const n = 48;
      const a = mod.alloc(n);
      const b = mod.alloc(64); // above `a`, so `a` cannot extend in place
      expect(b).not.toBe(0);

      const ua = new Uint8Array(mod.memory.buffer, a, n);
      for (let i = 0; i < n; i++) ua[i] = (i * 13 + 5) & 0xff;

      const a2 = mod.realloc(a, 8192);
      expect(a2).not.toBe(0);
      expect(a2).not.toBe(a); // had to relocate
      expect(a2 % 16).toBe(0);

      const ua2 = new Uint8Array(mod.memory.buffer, a2, n);
      for (let i = 0; i < n; i++) {
        expect(ua2[i], `byte ${i}`).toBe((i * 13 + 5) & 0xff);
      }
      mod.free(b);
      mod.free(a2);
    });

    it('realloc shrinks in place preserving the retained prefix', () => {
      const n = 512;
      const p = mod.alloc(n);
      const u = new Uint8Array(mod.memory.buffer, p, n);
      for (let i = 0; i < n; i++) u[i] = (i * 3) & 0xff;

      const p2 = mod.realloc(p, 64);
      expect(p2).not.toBe(0);
      expect(p2 % 16).toBe(0);

      const u2 = new Uint8Array(mod.memory.buffer, p2, 64);
      for (let i = 0; i < 64; i++) {
        expect(u2[i], `byte ${i}`).toBe((i * 3) & 0xff);
      }
    });

    it('realloc-move returns 0 on OOM, leaving the original block valid', () => {
      const a = mod.alloc(64);
      const u = new Uint8Array(mod.memory.buffer, a, 64);
      for (let i = 0; i < 64; i++) u[i] = (i + 9) & 0xff;
      const pin = mod.alloc(64); // pins `a` below top -> a grow must relocate
      expect(pin).not.toBe(0);

      // Push the heap top past 2 GiB so the relocating alloc() below overflows
      // the 4 GiB address space (1 GiB blocks stop ~3 GiB, leaving < 1 GiB).
      const big: number[] = [];
      for (let i = 0; i < 6; i++) {
        const p = mod.alloc(0x40000000);
        if (p === 0) break;
        big.push(p);
      }

      // A ~2 GiB relocation from a ~3 GiB heap top runs past 4 GiB -> OOM.
      const r = mod.realloc(a, 0x7ff00000); // must move; alloc() OOMs
      expect(r).toBe(0);

      // Original stays valid: a failed grow does not detach; bytes preserved.
      const uAfter = new Uint8Array(mod.memory.buffer, a, 64);
      for (let i = 0; i < 64; i++) {
        expect(uAfter[i], `byte ${i}`).toBe((i + 9) & 0xff);
      }
      for (const p of big) mod.free(p);
      mod.free(pin);
    });

    it('returns 0 on OOM at the wasm32 memory ceiling, then stays usable', () => {
      const big: number[] = [];
      let sawOom = false;
      for (let i = 0; i < 6; i++) {
        const p = mod.alloc(0x40000000); // 1 GiB; 6 x 1 GiB > 4 GiB ceiling
        if (p === 0) {
          sawOom = true;
          break;
        }
        big.push(p);
      }
      expect(sawOom).toBe(true);

      for (const p of big) mod.free(p);
      const p = mod.alloc(64);
      expect(p).not.toBe(0);
      expect(p % 16).toBe(0);
    });
  });
}
