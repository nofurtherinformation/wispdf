/**
 * Dictionary string store (Phase 1, deliverable P1.2 §2; ABI §4.4, ADR-002).
 *
 * A `utf8` column's strings are dictionary-encoded into three wasm buffers:
 *   - the column's own `i32[len]` **indices** (built in `column.ts`), plus this
 *     dictionary's shared:
 *   - **offsets** — `i32[count + 1]`, Arrow-style monotonic byte offsets,
 *     `offsets[0] == 0`, string `k` = `bytes[offsets[k] .. offsets[k+1])`;
 *   - **bytes** — `u8[offsets[count]]`, UTF-8 concatenation of the unique strings.
 *
 * Decode is **memoized per slot** on the JS side (ADR-002): each unique string
 * crosses the wasm→JS boundary at most once. The cache is keyed by *dictionary
 * identity* (a `WeakMap` on the `Dictionary` object) + slot index, so distinct
 * dictionaries never share decoded strings and the cache is GC'd with the dict.
 *
 * Unification (`unifyDictionaries`) is JS-side for Phase 1 — the wasm `unify_dict`
 * kernel arrives in Phase 2 (ABI §9, Agent D). It produces a merged unique list
 * plus per-slot index remaps so two columns can be compared under one dictionary.
 */

import type { MemoryContext } from './context.js';

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

/**
 * The shared dictionary buffers of a `utf8` column (ABI §4.4). Immutable: build
 * once, decode many. `count == 0` is legal (all-null / empty column): `offsets`
 * is still `[0]` and `bytesLen == 0`.
 */
export interface Dictionary {
  /** Number of unique strings. */
  readonly count: number;
  /** Byte offset of the `i32[count + 1]` offsets buffer. */
  readonly offsetsPtr: number;
  /** Byte offset of the `u8[bytesLen]` UTF-8 bytes buffer. */
  readonly bytesPtr: number;
  /** Total UTF-8 byte length (`offsets[count]`). */
  readonly bytesLen: number;
}

/** Per-dictionary decode memoization state (ADR-002). */
interface DecodeCache {
  /** `slots[k]` = decoded string for slot `k`, or `undefined` until first decode. */
  readonly slots: Array<string | undefined>;
  /** Boundary-crossing accounting (test-observable): a miss decodes, a hit reuses. */
  readonly stats: { hits: number; misses: number };
}

/** Decode caches keyed by `Dictionary` identity so they GC with the dictionary. */
const decodeCaches = new WeakMap<Dictionary, DecodeCache>();

function cacheFor(dict: Dictionary): DecodeCache {
  let entry = decodeCaches.get(dict);
  if (entry === undefined) {
    entry = { slots: new Array<string | undefined>(dict.count), stats: { hits: 0, misses: 0 } };
    decodeCaches.set(dict, entry);
  }
  return entry;
}

/**
 * Encode `uniques` (already-deduplicated strings) into fresh offsets + bytes
 * buffers in linear memory and return the {@link Dictionary} describing them.
 * Allocates all buffers *before* taking any view (a grow between alloc and write
 * would detach it, ADR-001).
 */
export function writeDictionary(ctx: MemoryContext, uniques: readonly string[]): Dictionary {
  const count = uniques.length;
  const encoded = uniques.map((s) => textEncoder.encode(s));
  let bytesLen = 0;
  for (const e of encoded) bytesLen += e.length;

  // Allocate both buffers first; then take views over the (possibly grown) buffer.
  const offsetsPtr = ctx.mod.alloc((count + 1) * 4);
  const bytesPtr = ctx.mod.alloc(bytesLen); // alloc(0) is a valid, un-dereferenced ptr

  const offsets = ctx.viewOf({ ptr: offsetsPtr, length: count + 1, dtype: 'i32' }) as Int32Array;
  const bytes = ctx.viewOf({ ptr: bytesPtr, length: bytesLen, dtype: 'u8' }) as Uint8Array;

  offsets[0] = 0;
  let acc = 0;
  for (let k = 0; k < count; k++) {
    const e = encoded[k]!;
    bytes.set(e, acc);
    acc += e.length;
    offsets[k + 1] = acc;
  }
  return { count, offsetsPtr, bytesPtr, bytesLen };
}

/**
 * Write a dictionary directly from raw UTF-8 bytes and Arrow-style offsets —
 * no JS string decode or encode (ABI §12 ingest note, ADR-002).
 *
 * `rawOffsets[k]..rawOffsets[k+1]` is the byte range of string `k` in
 * `rawBytes`; both arrays are already in our dictionaries' layout. This is a
 * pure bulk-copy: O(count + bytesLen), zero TextDecoder/TextEncoder calls.
 *
 * Allocates both wasm buffers before taking any view so a grow between the two
 * allocs cannot detach a live TypedArray (ADR-001).
 */
export function writeDictionaryFromRawBytes(
  ctx: MemoryContext,
  rawBytes: Uint8Array,
  rawOffsets: Int32Array,
  count: number,
): Dictionary {
  const bytesLen = count > 0 ? (rawOffsets[count] ?? 0) : 0;

  // Allocate both buffers before taking any view (ADR-001 grow-invalidation rule).
  const offsetsPtr = ctx.mod.alloc((count + 1) * 4);
  const bytesPtr   = ctx.mod.alloc(bytesLen); // alloc(0) is valid per ABI §3

  const offsets = ctx.viewOf({ ptr: offsetsPtr, length: count + 1, dtype: 'i32' }) as Int32Array;
  offsets.set(rawOffsets.subarray(0, count + 1));

  if (bytesLen > 0) {
    const bytes = ctx.viewOf({ ptr: bytesPtr, length: bytesLen, dtype: 'u8' }) as Uint8Array;
    bytes.set(rawBytes.subarray(0, bytesLen));
  }

  return { count, offsetsPtr, bytesPtr, bytesLen };
}

/**
 * Decode dictionary slot `slot` to a string, memoized (ADR-002). First call for a
 * slot reads its UTF-8 bytes across the boundary (a *miss*); later calls reuse the
 * cached string (a *hit*). `slot` must be in `[0, dict.count)`.
 */
export function decodeSlot(ctx: MemoryContext, dict: Dictionary, slot: number): string {
  const entry = cacheFor(dict);
  const hit = entry.slots[slot];
  if (hit !== undefined) {
    entry.stats.hits++;
    return hit;
  }
  entry.stats.misses++;
  const offsets = ctx.viewOf({ ptr: dict.offsetsPtr, length: dict.count + 1, dtype: 'i32' }) as Int32Array;
  const start = offsets[slot]!;
  const end = offsets[slot + 1]!;
  let s: string;
  if (end <= start) {
    s = '';
  } else {
    const bytes = ctx.viewOf({ ptr: dict.bytesPtr, length: dict.bytesLen, dtype: 'u8' }) as Uint8Array;
    s = textDecoder.decode(bytes.subarray(start, end));
  }
  entry.slots[slot] = s;
  return s;
}

/** Decode every slot of `dict` into a `string[]` (memoized per slot). */
export function decodeDictionary(ctx: MemoryContext, dict: Dictionary): string[] {
  const out = new Array<string>(dict.count);
  for (let i = 0; i < dict.count; i++) out[i] = decodeSlot(ctx, dict, i);
  return out;
}

/** Decode-cache accounting for `dict` (`{hits:0,misses:0}` if never decoded). */
export function decodeStats(dict: Dictionary): { hits: number; misses: number } {
  const entry = decodeCaches.get(dict);
  return entry === undefined ? { hits: 0, misses: 0 } : { ...entry.stats };
}

/** Result of unifying two dictionaries into a single merged one (JS-side). */
export interface DictUnifyResult {
  /** Merged unique strings; index `j` is a slot in the merged dictionary. */
  readonly merged: string[];
  /** `remapA[i]` = merged slot of dictionary-A slot `i` (`Int32Array`, ABI indices). */
  readonly remapA: Int32Array;
  /** `remapB[i]` = merged slot of dictionary-B slot `i`. */
  readonly remapB: Int32Array;
}

/**
 * Unify two dictionaries into one merged unique list plus per-slot index remaps
 * (JS-side; the wasm `unify_dict` kernel is Phase 2). Value-preserving by
 * construction: `merged[remapA[i]] === decode(dictA, i)` for every `i`. The
 * merged order is A's slots in order, then B's not-yet-seen slots.
 */
export function unifyDictionaries(
  ctx: MemoryContext,
  dictA: Dictionary,
  dictB: Dictionary,
): DictUnifyResult {
  const a = decodeDictionary(ctx, dictA);
  const b = decodeDictionary(ctx, dictB);
  const merged: string[] = [];
  const index = new Map<string, number>();
  const intern = (s: string): number => {
    let j = index.get(s);
    if (j === undefined) {
      j = merged.length;
      merged.push(s);
      index.set(s, j);
    }
    return j;
  };
  const remapA = new Int32Array(a.length);
  const remapB = new Int32Array(b.length);
  for (let i = 0; i < a.length; i++) remapA[i] = intern(a[i]!);
  for (let i = 0; i < b.length; i++) remapB[i] = intern(b[i]!);
  return { merged, remapA, remapB };
}

/** Free a dictionary's buffers and drop them from the view registry. */
export function freeDictionary(ctx: MemoryContext, dict: Dictionary): void {
  ctx.viewOf.forget({ ptr: dict.offsetsPtr, length: dict.count + 1, dtype: 'i32' });
  ctx.viewOf.forget({ ptr: dict.bytesPtr, length: dict.bytesLen, dtype: 'u8' });
  ctx.mod.free(dict.offsetsPtr);
  ctx.mod.free(dict.bytesPtr);
}
