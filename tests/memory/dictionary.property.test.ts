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
