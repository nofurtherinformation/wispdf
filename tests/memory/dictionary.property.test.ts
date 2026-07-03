import { describe, it, expect, beforeAll } from 'vitest';
import fc from 'fast-check';
import { ctxForTest } from './helper.js';
import type { MemoryContext } from '../../src/memory/context.js';
import {
  createColumn,
  columnToArray,
  freeColumn,
  decodeDictionary,
  decodeStats,
  unifyDictionaries,
  writeDictionary,
  writeDictionaryFromRawBytes,
  freeDictionary,
} from '../../src/memory/index.js';

// P1.2 dictionary-store gate (ABI §4.4, ADR-002):
//   - decode(encode(x)) == x for random string arrays with nulls/duplicates,
//   - unify preserves values (merged[remap[i]] == original slot i),
//   - per-slot decode memoization: each unique string crosses the boundary once.

let ctx: MemoryContext;
beforeAll(async () => {
  ctx = await ctxForTest(false);
});

const strArb = fc.oneof(
  fc.constant(null),
  fc.string(),
  fc.fullUnicodeString(),
  fc.constantFrom('', 'a', 'a', 'b', 'foo', 'bar', '😀', '你好', 'a', 'foo'),
);

describe('dictionary build / decode round-trip', () => {
  it('decode(encode(x)) == x and the dict holds the unique strings in first-seen order', () => {
    fc.assert(
      fc.property(fc.array(strArb, { maxLength: 50 }), (input) => {
        const col = createColumn(ctx, 'utf8', input);
        try {
          // Round-trip identity.
          expect(columnToArray(ctx, col)).toEqual(input.map((v) => (v === undefined ? null : v)));

          // The dictionary contains exactly the distinct non-null strings,
          // in order of first appearance, with no duplicates.
          const seen: string[] = [];
          const set = new Set<string>();
          for (const v of input) {
            if (v !== null && !set.has(v)) {
              set.add(v);
              seen.push(v);
            }
          }
          const dictStrings = decodeDictionary(ctx, col.dict!);
          expect(dictStrings).toEqual(seen);
          expect(col.dict!.count).toBe(seen.length);
        } finally {
          freeColumn(ctx, col);
        }
      }),
      { numRuns: 250 },
    );
  });
});

describe('dictionary decode memoization (ADR-002)', () => {
  it('each unique string is decoded across the boundary at most once', () => {
    fc.assert(
      fc.property(fc.array(strArb, { maxLength: 60 }), (input) => {
        const col = createColumn(ctx, 'utf8', input);
        try {
          const dict = col.dict!;
          const nonNull = input.filter((v) => v !== null).length;

          // Fresh dictionary: nothing decoded yet.
          expect(decodeStats(dict)).toEqual({ hits: 0, misses: 0 });

          // First full decode: every unique slot missed exactly once; every other
          // non-null reference is a cache hit.
          columnToArray(ctx, col);
          const after1 = decodeStats(dict);
          expect(after1.misses).toBe(dict.count);
          expect(after1.hits).toBe(nonNull - dict.count);

          // Second full decode: no new boundary crossings — all hits.
          columnToArray(ctx, col);
          const after2 = decodeStats(dict);
          expect(after2.misses).toBe(dict.count); // unchanged
          expect(after2.hits).toBe(after1.hits + nonNull);
        } finally {
          freeColumn(ctx, col);
        }
      }),
      { numRuns: 150 },
    );
  });
});

describe('dictionary unification (JS-side; wasm unify_dict is Phase 2)', () => {
  it('merges two dictionaries preserving every value and de-duplicating', () => {
    fc.assert(
      fc.property(
        fc.array(strArb, { maxLength: 40 }),
        fc.array(strArb, { maxLength: 40 }),
        (aIn, bIn) => {
          const colA = createColumn(ctx, 'utf8', aIn);
          const colB = createColumn(ctx, 'utf8', bIn);
          try {
            const dictA = decodeDictionary(ctx, colA.dict!);
            const dictB = decodeDictionary(ctx, colB.dict!);
            const { merged, remapA, remapB } = unifyDictionaries(ctx, colA.dict!, colB.dict!);

            // Value preservation: remap of each source slot points at the same string.
            for (let i = 0; i < dictA.length; i++) expect(merged[remapA[i]!]).toBe(dictA[i]);
            for (let i = 0; i < dictB.length; i++) expect(merged[remapB[i]!]).toBe(dictB[i]);

            // Merged is the deduplicated union of both dictionaries.
            expect(new Set(merged)).toEqual(new Set([...dictA, ...dictB]));
            expect(merged.length).toBe(new Set(merged).size); // no duplicates
          } finally {
            freeColumn(ctx, colA);
            freeColumn(ctx, colB);
          }
        },
      ),
      { numRuns: 150 },
    );
  });
});

// ---------------------------------------------------------------------------
// CP.1 — writeDictionaryFromRawBytes (ABI §12)
// ---------------------------------------------------------------------------

describe('writeDictionaryFromRawBytes — identical to writeDictionary (CP.1)', () => {
  const enc = new TextEncoder();
  const dec = new TextDecoder();

  /**
   * Build raw Arrow-style bytes + offsets for a string array, then verify that
   * writeDictionaryFromRawBytes produces the same decodable dictionary as
   * writeDictionary(ctx, strings).
   */
  it('produces byte-identical dictionaries to writeDictionary for random string arrays', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.oneof(fc.string(), fc.fullUnicodeString(), fc.constant(''), fc.constant('hello')),
          { maxLength: 60 },
        ),
        (strings) => {
          // Build raw bytes + offsets in the same layout as Arrow plain UTF-8
          const encoded = strings.map((s) => enc.encode(s));
          let totalBytes = 0;
          for (const e of encoded) totalBytes += e.length;
          const rawBytes   = new Uint8Array(totalBytes);
          const rawOffsets = new Int32Array(strings.length + 1);
          let pos = 0;
          for (let k = 0; k < strings.length; k++) {
            rawOffsets[k] = pos;
            rawBytes.set(encoded[k]!, pos);
            pos += encoded[k]!.length;
          }
          rawOffsets[strings.length] = pos;

          // Build via old path (string → TextEncoder)
          const dictOld = writeDictionary(ctx, strings);
          // Build via new path (raw bytes → bulk-copy)
          const dictNew = writeDictionaryFromRawBytes(ctx, rawBytes, rawOffsets, strings.length);

          try {
            expect(dictNew.count).toBe(dictOld.count);
            expect(dictNew.bytesLen).toBe(dictOld.bytesLen);
            // Slot-by-slot: every slot decodes to the same string
            const decodedOld = decodeDictionary(ctx, dictOld);
            const decodedNew = decodeDictionary(ctx, dictNew);
            expect(decodedNew).toEqual(decodedOld);
          } finally {
            freeDictionary(ctx, dictOld);
            freeDictionary(ctx, dictNew);
          }
        },
      ),
      { numRuns: 200 },
    );
  });

  it('handles empty string array (count=0)', () => {
    const dict = writeDictionaryFromRawBytes(ctx, new Uint8Array(0), new Int32Array([0]), 0);
    try {
      expect(dict.count).toBe(0);
      expect(dict.bytesLen).toBe(0);
    } finally {
      freeDictionary(ctx, dict);
    }
  });

  it('handles all-empty-string array', () => {
    const rawBytes   = new Uint8Array(0);
    const rawOffsets = new Int32Array([0, 0, 0, 0]);
    const dict = writeDictionaryFromRawBytes(ctx, rawBytes, rawOffsets, 3);
    const expected = writeDictionary(ctx, ['', '', '']);
    try {
      expect(dict.count).toBe(expected.count);
      expect(decodeDictionary(ctx, dict)).toEqual(decodeDictionary(ctx, expected));
    } finally {
      freeDictionary(ctx, dict);
      freeDictionary(ctx, expected);
    }
  });
});
