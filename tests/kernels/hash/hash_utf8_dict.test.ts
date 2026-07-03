/**
 * Tests for `hash_utf8_dict` kernel (ABI §12, v2.2 amendment).
 *
 * Covers:
 *   - Equal UTF-8 bytes in independently built dictionaries → identical hashes
 *     (THE regression test: this is what unification-free joins depend on)
 *   - Empty string → fixed, non-zero hash
 *   - Unicode strings (multi-byte sequences)
 *   - Long strings (>256 bytes)
 *   - Distinct strings → distinct hashes (no accidental collisions in test data)
 *   - Scalar and SIMD builds produce bit-identical outputs
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import type { HashExports } from '../../../src/kernels/hash/index.js';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = fileURLToPath(new URL('../../../wasm/dist/', import.meta.url));

const BUILDS: ReadonlyArray<{ label: string; simd: boolean }> = [
  { label: 'scalar', simd: false },
  { label: 'simd',   simd: true  },
];

async function loadExports(simd: boolean): Promise<HashExports> {
  const fileName = simd ? 'simd.wasm' : 'scalar.wasm';
  const bytes = await readFile(join(WASM_DIR, fileName));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports as unknown as HashExports;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const textEncoder = new TextEncoder();

/** Build and write a dictionary into WASM memory from an array of strings. */
function writeDictToWasm(
  ex: HashExports,
  strs: string[],
): { offsetsPtr: number; bytesPtr: number; count: number } {
  const encoded = strs.map((s) => textEncoder.encode(s));
  const totalBytes = encoded.reduce((acc, e) => acc + e.length, 0);
  const count = strs.length;

  const offsetsPtr = ex.alloc((count + 1) * 4);
  const bytesPtr   = ex.alloc(Math.max(totalBytes, 1));

  const offsets = new Int32Array(ex.memory.buffer, offsetsPtr, count + 1);
  const bytes   = new Uint8Array(ex.memory.buffer, bytesPtr,   Math.max(totalBytes, 1));

  offsets[0] = 0;
  let acc = 0;
  for (let k = 0; k < count; k++) {
    const e = encoded[k]!;
    bytes.set(e, acc);
    acc += e.length;
    offsets[k + 1] = acc;
  }
  return { offsetsPtr, bytesPtr, count };
}

/** Call hash_utf8_dict and return the output as bigint[]. */
function runHashUtf8Dict(
  ex: HashExports,
  strs: string[],
): bigint[] {
  const { offsetsPtr, bytesPtr, count } = writeDictToWasm(ex, strs);
  const outPtr = ex.alloc(Math.max(count * 8, 1));
  try {
    ex.hash_utf8_dict(offsetsPtr, bytesPtr, count, outPtr);
    return count === 0
      ? []
      : Array.from(new BigInt64Array(ex.memory.buffer, outPtr, count));
  } finally {
    ex.free(offsetsPtr);
    ex.free(bytesPtr);
    ex.free(outPtr);
  }
}

// ---------------------------------------------------------------------------
// Tests — both builds
// ---------------------------------------------------------------------------

describe('hash_utf8_dict kernel', () => {
  for (const build of BUILDS) {
    describe(`[${build.label}]`, () => {
      let ex: HashExports;

      beforeAll(async () => {
        ex = await loadExports(build.simd);
      });

      // ── Export exists ──────────────────────────────────────────────────────

      it('exports hash_utf8_dict', () => {
        expect(typeof ex.hash_utf8_dict).toBe('function');
      });

      // ── Empty dict (count = 0) ─────────────────────────────────────────────

      it('empty dict: no output (count=0 is a no-op)', () => {
        const hashes = runHashUtf8Dict(ex, []);
        expect(hashes).toEqual([]);
      });

      // ── Empty string → non-zero hash ──────────────────────────────────────

      it('empty string hashes to a non-zero value', () => {
        const [h] = runHashUtf8Dict(ex, ['']);
        expect(h).not.toBe(0n);
      });

      // ── Equal bytes ↔ equal hashes (the regression-risk path) ─────────────

      it('same strings built into separate dicts produce identical per-slot hashes', () => {
        // Simulate two sides of a join with DIFFERENT slot orders (the hard case):
        // left dict:  ["cat", "dog", "fish"]  at slots [0, 1, 2]
        // right dict: ["fish", "cat", "dog"]  at slots [0, 1, 2]
        // Expected: hash(left[0]) == hash(right[1])  ("cat")
        //           hash(left[1]) == hash(right[2])  ("dog")
        //           hash(left[2]) == hash(right[0])  ("fish")
        const leftStrs  = ['cat', 'dog', 'fish'];
        const rightStrs = ['fish', 'cat', 'dog'];
        const lHashes = runHashUtf8Dict(ex, leftStrs);
        const rHashes = runHashUtf8Dict(ex, rightStrs);

        expect(lHashes[0]).toBe(rHashes[1]); // "cat" == "cat"
        expect(lHashes[1]).toBe(rHashes[2]); // "dog" == "dog"
        expect(lHashes[2]).toBe(rHashes[0]); // "fish" == "fish"
      });

      it('same string appearing twice in different dicts hashes equally', () => {
        const h1 = runHashUtf8Dict(ex, ['hello'])[0]!;
        const h2 = runHashUtf8Dict(ex, ['hello'])[0]!;
        expect(h1).toBe(h2);
      });

      // ── Distinct strings → distinct hashes ────────────────────────────────

      it('distinct strings produce distinct hashes', () => {
        const strs = ['alpha', 'beta', 'gamma', 'delta', 'epsilon'];
        const hashes = runHashUtf8Dict(ex, strs);
        const unique = new Set(hashes.map(String));
        expect(unique.size).toBe(strs.length);
      });

      it('empty string hash differs from single-byte string hashes', () => {
        const strs = ['', 'a', 'b', '\0'];
        const hashes = runHashUtf8Dict(ex, strs);
        const unique = new Set(hashes.map(String));
        expect(unique.size).toBe(strs.length);
      });

      // ── Unicode strings ────────────────────────────────────────────────────

      it('multi-byte unicode strings hash correctly (equal bytes ↔ equal hashes)', () => {
        const unicodeStrs = ['héllo', '日本語', '🎉🎊', 'ñoño', 'العربية'];
        const h1 = runHashUtf8Dict(ex, unicodeStrs);
        const h2 = runHashUtf8Dict(ex, [...unicodeStrs]); // same strings, fresh dict
        expect(h1).toEqual(h2);
      });

      it('unicode strings in reversed order in the other dict still match by content', () => {
        const strs   = ['héllo', '日本語', '🎉🎊'];
        const strRev = ['🎉🎊', 'héllo', '日本語'];
        const lH = runHashUtf8Dict(ex, strs);
        const rH = runHashUtf8Dict(ex, strRev);
        // héllo  at lH[0] == rH[1]
        // 日本語 at lH[1] == rH[2]
        // 🎉🎊   at lH[2] == rH[0]
        expect(lH[0]).toBe(rH[1]);
        expect(lH[1]).toBe(rH[2]);
        expect(lH[2]).toBe(rH[0]);
      });

      // ── Long strings ───────────────────────────────────────────────────────

      it('long strings (>256 bytes) hash deterministically and equally across dicts', () => {
        const long = 'x'.repeat(1000);
        const h1 = runHashUtf8Dict(ex, [long])[0]!;
        const h2 = runHashUtf8Dict(ex, [long])[0]!;
        expect(h1).toBe(h2);
        expect(h1).not.toBe(0n);
      });

      it('strings differing only in the last byte produce different hashes', () => {
        const base = 'a'.repeat(500);
        const strs = [base + 'x', base + 'y'];
        const hashes = runHashUtf8Dict(ex, strs);
        expect(hashes[0]).not.toBe(hashes[1]);
      });

      // ── Multi-slot dict roundtrip ──────────────────────────────────────────

      it('single-element dict roundtrips correctly', () => {
        const strs = ['only'];
        const hashes = runHashUtf8Dict(ex, strs);
        expect(hashes.length).toBe(1);
        expect(hashes[0]).not.toBe(0n);
      });
    });
  }

  // ── Scalar == SIMD (bit-identical across builds) ────────────────────────────

  describe('scalar == simd (bit-identical outputs)', () => {
    let scalarEx: HashExports;
    let simdEx: HashExports;

    beforeAll(async () => {
      [scalarEx, simdEx] = await Promise.all([
        loadExports(false),
        loadExports(true),
      ]);
    });

    it('single string: both builds return the same hash', () => {
      const strs = ['test string'];
      const hs = runHashUtf8Dict(scalarEx, strs);
      const hm = runHashUtf8Dict(simdEx,   strs);
      expect(hs).toEqual(hm);
    });

    it('multiple strings including empty and unicode: both builds agree', () => {
      const strs = ['', 'a', 'hello world', '日本語', '🎉', 'x'.repeat(300)];
      const hs = runHashUtf8Dict(scalarEx, strs);
      const hm = runHashUtf8Dict(simdEx,   strs);
      expect(hs).toEqual(hm);
    });

    it('large dict (1000 distinct strings): both builds produce identical arrays', () => {
      const strs = Array.from({ length: 1000 }, (_, i) => `item_${i}_${'x'.repeat(i % 64)}`);
      const hs = runHashUtf8Dict(scalarEx, strs);
      const hm = runHashUtf8Dict(simdEx,   strs);
      expect(hs).toEqual(hm);
    });
  });
});
