/**
 * contracts/memory.d.ts — Memory core interface (Phase 1)
 *
 * STATUS: v1 final + v2 deltas. Written by P1.1 (arena allocator, dual wasm
 * builds, loader, viewOf layer) and finalized by P1.2 (dtype registry, validity
 * bitmap, column representation, dictionary string store, zero-copy slice). This
 * is the typed surface the kernel (Phase 2), expression (Phase 3), and frame
 * (Phase 3) layers build on.
 *
 * v2 deltas (bead dataframe-dh9.1; ADR-009 i64, ADR-010 temporals): `DType` gains
 * `'i64' | 'date32' | 'timestamp'`; `ViewDType`/`ColumnView`/`TypedArrayCtor` gain
 * `BigInt64Array`; `Cell`/`ColumnInput` gain `bigint`; `Column` gains optional `tz`
 * metadata; `DTypeInfo.wasm` is now the *physical* kernel token (differs from `name`
 * for temporals). No memory-layer function signatures changed.
 *
 * It mirrors the runtime in `src/memory/` and encodes the ABI guarantees from
 * `contracts/wasm-abi.md` (§2 memory ownership + generation counter, §3
 * allocator exports, §4 buffer conventions, §9 export list) and the ADRs
 * (ADR-001 single viewOf, ADR-002 Arrow columnar + dict encoding, ADR-004 dual
 * feature-detected builds).
 *
 * Companion (read-only, authoritative): contracts/wasm-abi.md, contracts/dtypes.md.
 */

// ===========================================================================
// Loader (ADR-004 — dual builds, feature-detected)
// ===========================================================================

/** Raw exports of the memory-core wasm module (ABI §9, Phase 1). */
export interface WasmExports {
  /** The module's single linear memory. ALL column bytes live here (ADR-001). */
  readonly memory: WebAssembly.Memory;
  /**
   * `alloc(size) -> ptr`. Returns a 16-byte-aligned byte offset, or `0` on OOM
   * (a failed `memory.grow`). `alloc(0)` returns a valid aligned pointer that
   * must not be dereferenced (ABI §3).
   */
  alloc(size: number): number;
  /** `free(ptr)`. `free(0)` is a no-op. Double-free is undefined (ABI §3). */
  free(ptr: number): void;
  /**
   * `realloc(ptr, newSize) -> ptr`. 16-byte-aligned; preserves the first
   * `min(old, new)` bytes; returns `0` on OOM with the original block left
   * valid. `realloc(0, n)` is equivalent to `alloc(n)` (ABI §3).
   */
  realloc(ptr: number, newSize: number): number;
  /**
   * `mem_generation() -> i32`. A monotonically increasing counter whose value
   * changes on every successful `memory.grow` (ABI §2). `viewOf` compares this
   * against its cache to know when views must be rebuilt.
   */
  mem_generation(): number;
}

/** A loaded memory-core module plus which build was selected. */
export interface WasmMemoryModule extends WasmExports {
  /** `true` if the SIMD128 build (`simd.wasm`) was loaded, else scalar. */
  readonly simd: boolean;
}

/** Options for {@link loadWasmModule}. */
export interface LoadOptions {
  /** Force a build; default auto-detects SIMD via {@link detectSimd}. */
  simd?: boolean;
  /**
   * Where `scalar.wasm` / `simd.wasm` live.
   *  - Node: a directory path, or a `file:` / directory `URL`.
   *  - Browser: a base `URL` (or URL string) to fetch the two files under.
   * Default: resolved relative to the loader module (binaries ship next to the
   * JS bundle).
   */
  wasmDir?: string | URL;
}

/** True iff the current runtime supports wasm SIMD128 (ADR-004). */
export declare function detectSimd(): boolean;

/**
 * Feature-detect, load, and instantiate the memory-core wasm module. Intended
 * to run once per page/process; the caller caches the returned module and
 * derives a single {@link ViewOf} from it.
 */
export declare function loadWasmModule(
  opts?: LoadOptions,
): Promise<WasmMemoryModule>;

// ===========================================================================
// viewOf layer (ADR-001 / ABI §2 — the ONLY sanctioned TypedArray accessor)
// ===========================================================================

/**
 * Storage dtypes whose data buffer maps to a numeric `TypedArray` (dtypes.md
 * §1). `bool` is `u8` storage (0/1). NOTE: `utf8` is not here — a `utf8` column
 * is three buffers (i32 indices + i32 offsets + u8 bytes, ABI §4.4); P1.2 adds
 * the higher-level column/dictionary descriptors.
 *
 * v2 (ADR-009): `'i64'` maps to `BigInt64Array`. Temporal logical dtypes have no
 * distinct view — `date32` uses the `'i32'` view, `timestamp` uses `'i64'`.
 */
export type ViewDType = 'f64' | 'f32' | 'i32' | 'u32' | 'u8' | 'bool' | 'i64';

/** Location + shape of a column buffer inside linear memory. */
export interface ColumnBuffer {
  /** Byte offset into `memory.buffer`; 16-byte aligned (ABI §3). */
  readonly ptr: number;
  /** Element count (NOT bytes; ABI §4.3). */
  readonly length: number;
  /** Determines the `TypedArray` kind. */
  readonly dtype: ViewDType;
}

/** The concrete `TypedArray` kinds a column view can be. */
export type ColumnView =
  | Float64Array
  | Float32Array
  | Int32Array
  | Uint32Array
  | Uint8Array
  | BigInt64Array; // v2 (ADR-009): i64 / timestamp physical storage

/**
 * The single `viewOf` accessor (ADR-001). Caches `(generation, view)` per
 * registered column, checks `mem_generation()` before every use, and rebuilds
 * ALL registered views over the current `memory.buffer` on a mismatch. No raw
 * `TypedArray` may be cached anywhere else; callers re-call `viewOf` each use
 * and never hold the returned view across a call that may grow memory.
 */
export interface ViewOf {
  /** Live view of `col` over the current buffer (rebuilt if memory grew). */
  (col: ColumnBuffer): ColumnView;
  /** Generation the cache is currently synced to. */
  generation(): number;
  /** Stop tracking `col`. */
  forget(col: ColumnBuffer): void;
  /** Drop all tracked columns and cached views. */
  clear(): void;
}

/** Build the memory context's single `viewOf` over a module's memory + counter. */
export declare function createViewOf(
  mod: Pick<WasmMemoryModule, 'memory' | 'mem_generation'>,
): ViewOf;

// ===========================================================================
// MemoryContext — allocator + the single viewOf, bundled
// ===========================================================================

/**
 * The allocator (`ctx.mod`) plus the one sanctioned `viewOf` accessor over its
 * linear memory. Every column / dictionary operation takes a context: it
 * allocates through `ctx.mod` and reaches bytes only through `ctx.viewOf`. There
 * is exactly one context per loaded module (one memory, one generation counter).
 */
export interface MemoryContext {
  readonly mod: WasmMemoryModule;
  readonly viewOf: ViewOf;
}

/** Build the single {@link MemoryContext} for a loaded module. */
export declare function createMemoryContext(mod: WasmMemoryModule): MemoryContext;

// ===========================================================================
// Dtype registry (dtypes.md §1) — storage size, TypedArray, kernel-name token
// ===========================================================================

/**
 * The column dtypes. v1 (dtypes.md §1): `f64 f32 i32 u32 bool utf8` (`utf8` is
 * dictionary-encoded, ABI §4.4). v2 adds:
 *  - `'i64'`      — signed 64-bit int, `BigInt64Array` storage (ADR-009).
 *  - `'date32'`   — logical: days since epoch; **physical i32** (ADR-010).
 *  - `'timestamp'`— logical: ms since epoch, always UTC; **physical i64**;
 *                   optional {@link Column.tz} metadata (ADR-010).
 * For temporals the kernel token ({@link DTypeInfo.wasm}) is the physical dtype,
 * not the name (`date32`→`i32`, `timestamp`→`i64`).
 */
export type DType =
  | 'f64'
  | 'f32'
  | 'i32'
  | 'u32'
  | 'bool'
  | 'utf8'
  | 'i64'
  | 'date32'
  | 'timestamp';

/** `TypedArray` constructors a column data / auxiliary buffer can map to. */
export type TypedArrayCtor =
  | Float64ArrayConstructor
  | Float32ArrayConstructor
  | Int32ArrayConstructor
  | Uint32ArrayConstructor
  | Uint8ArrayConstructor
  | BigInt64ArrayConstructor; // v2 (ADR-009): i64 / timestamp

/** Static description of one dtype used by column creation and (P2+) kernels. */
export interface DTypeInfo {
  /** The dtype this describes. */
  readonly name: DType;
  /**
   * Bytes per stored element. `utf8` = 4 (the `i32` index into its dictionary).
   * v2: `i64`/`timestamp` = 8, `date32` = 4.
   */
  readonly size: number;
  /**
   * `viewOf` dtype for this column's *data* buffer (`utf8` → its `i32` indices).
   * v2: `i64` → `'i64'` (`BigInt64Array`); `date32` → `'i32'`; `timestamp` → `'i64'`.
   */
  readonly view: ViewDType;
  /** `TypedArray` constructor matching {@link view} (for staging copies). */
  readonly ctor: TypedArrayCtor;
  /**
   * Kernel-name dtype token (ABI §6) — the **physical** dtype. Equal to
   * {@link name} for every v1 dtype and for `i64`. **v2 (ADR-010): temporals
   * diverge** — `date32` → `'i32'`, `timestamp` → `'i64'` — encoding the
   * logical→physical registry that lets temporals reuse i32/i64 kernels.
   */
  readonly wasm: string;
  /** True for `f64`/`f32`: `NaN`/`±inf` are valid *values*, never nulls (dtypes.md §4). Integers (incl. `i64`, `date32`, `timestamp`) are `false`. */
  readonly float: boolean;
}

/** The dtype registry: `DTYPES[dtype]` → its {@link DTypeInfo}. */
export declare const DTYPES: Record<DType, DTypeInfo>;

/** Descriptor for `dtype`. Throws on an unknown dtype. */
export declare function dtypeInfo(dtype: DType): DTypeInfo;

// ===========================================================================
// Validity bitmap (ABI §4.1) — Arrow LSB-first, 1 = valid, 0 = null
// ===========================================================================

/** Bytes needed for a `len`-bit validity bitmap: `ceil(len / 8)`. */
export declare function validityBytes(len: number): number;

/** True iff bit `i` is set (element valid) in an LSB-first bitmap. */
export declare function getBit(bitmap: Uint8Array, i: number): boolean;

/** Set bit `i` (mark element valid). */
export declare function setBit(bitmap: Uint8Array, i: number): void;

/** Clear bit `i` (mark element null). */
export declare function clearBit(bitmap: Uint8Array, i: number): void;

// ===========================================================================
// Column representation (ABI §4) + zero-copy slice
// ===========================================================================

/**
 * A column descriptor over buffers in wasm linear memory (ABI §4). JS never owns
 * the bytes (ADR-001); it holds this descriptor and reaches data through `viewOf`.
 *
 * ## Offset convention (zero-copy slice)
 * A slice shares the parent's buffers; no bytes move. Two offset kinds:
 *  - **data** is byte-addressable, so a slice bakes its start into `dataPtr`
 *    (`parent.dataPtr + start * dtype.size`). **`dataPtr` is exactly the base
 *    pointer a kernel receives** — byte-offset math is done here, so a Phase-2
 *    kernel takes `(dataPtr, length)` unchanged. Whole-element offsets preserve
 *    the natural alignment `alloc` guarantees.
 *  - **validity** is *bit*-addressed, so a slice keeps the parent's `validityPtr`
 *    and records `validityBitOffset` (the bit index of element 0; `0` for a root).
 *    Element `i`'s validity bit is `validityBitOffset + i`. A kernel consuming a
 *    sliced column's validity must honor this bit offset (or the memory layer
 *    realigns first); the common root case (`validityBitOffset == 0`) passes
 *    `validityPtr` directly. `validityPtr == 0` always means all-valid (ABI §4.1).
 */
export interface Column {
  /** Storage dtype (dtypes.md §1). */
  readonly dtype: DType;
  /** Element count of this (possibly sliced) column. */
  readonly length: number;
  /** Base byte offset of element 0's data (slice start already baked in). */
  readonly dataPtr: number;
  /** Validity bitmap pointer, or `0` for an all-valid column (ABI §4.1). */
  readonly validityPtr: number;
  /** Bit index of element 0 within the bitmap at `validityPtr` (0 for roots). */
  readonly validityBitOffset: number;
  /** The shared dictionary for a `utf8` column; `null` for every other dtype. */
  readonly dict: Dictionary | null;
  /**
   * v2 (ADR-010): optional IANA timezone metadata for a `timestamp` column
   * (e.g. `'America/New_York'`, `'UTC'`, `'+05:30'`). Stored values are **always
   * UTC ms**; `tz` affects only display and tz-aware `dt` accessors (Arrow model),
   * never the physical value. `null`/absent → UTC. Always `null` for every
   * non-`timestamp` dtype (`date32` is tz-independent).
   */
  readonly tz?: string | null;
  /** True if this column owns its buffers (a root); a slice owns nothing. */
  readonly owned: boolean;
}

/**
 * JS value shapes accepted by {@link createColumn}, per dtype. v2 (ADR-009):
 * `i64`/`timestamp` accept `bigint` or safe-integer `number` (a non-integer or
 * out-of-safe-range `number` throws); `date32` accepts a `number` of days.
 */
export type ColumnInput =
  | ArrayLike<number | null | undefined>
  | ArrayLike<boolean | null | undefined>
  | ArrayLike<string | null | undefined>
  | ArrayLike<bigint | null | undefined>; // v2: i64 / timestamp

/**
 * One decoded column cell: a value, or `null` for a null slot. v2 (ADR-009):
 * `i64`/`timestamp` cells are `bigint`.
 */
export type Cell = number | boolean | string | bigint | null;

/**
 * Build a column from JS `values` for `dtype`. A matching `TypedArray` takes the
 * bulk-copy fast path (no nulls, `validityPtr == 0`); a plain array takes the slow
 * path, detecting `null`/`undefined` (only these are nulls — a `NaN` is a *value*,
 * dtypes.md §4) and building the validity bitmap.
 */
export declare function createColumn(
  ctx: MemoryContext,
  dtype: DType,
  values: ColumnInput,
): Column;

/**
 * Export a column back to a JS array, null slots as `null`. `f64`/`f32` `NaN`s
 * round-trip as values; `bool` yields `boolean`; `utf8` yields memoized-decoded
 * strings.
 */
export declare function columnToArray(ctx: MemoryContext, col: Column): Cell[];

/**
 * Zero-copy slice `[start, end)` sharing the parent's buffers. `start`/`end` clamp
 * to `[0, length]`; `end < start` yields an empty slice. Slice-of-slice composes.
 * See {@link Column}'s offset convention.
 */
export declare function sliceColumn(col: Column, start: number, end: number): Column;

/**
 * Free a column's owned buffers (and, for `utf8`, its dictionary) and drop them
 * from the view registry. A no-op for slices (`owned === false`).
 */
export declare function freeColumn(ctx: MemoryContext, col: Column): void;

// ===========================================================================
// Dictionary string store (ABI §4.4, ADR-002)
// ===========================================================================

/**
 * The shared dictionary buffers of a `utf8` column (ABI §4.4): `i32[count+1]`
 * offsets + `u8[bytesLen]` UTF-8 bytes. String `k` = `bytes[offsets[k]..offsets[k+1])`.
 * `count == 0` is legal (all-null / empty); `offsets` is still `[0]`.
 */
export interface Dictionary {
  readonly count: number;
  readonly offsetsPtr: number;
  readonly bytesPtr: number;
  readonly bytesLen: number;
}

/** Result of unifying two dictionaries into one merged one (JS-side). */
export interface DictUnifyResult {
  /** Merged unique strings; index `j` is a slot in the merged dictionary. */
  readonly merged: string[];
  /** `remapA[i]` = merged slot of dictionary-A slot `i`. */
  readonly remapA: Int32Array;
  /** `remapB[i]` = merged slot of dictionary-B slot `i`. */
  readonly remapB: Int32Array;
}

/** Encode already-deduplicated `uniques` into fresh offsets + bytes buffers. */
export declare function writeDictionary(
  ctx: MemoryContext,
  uniques: readonly string[],
): Dictionary;

/**
 * Decode dictionary slot `slot` to a string, memoized per (dictionary, slot)
 * (ADR-002): each unique string crosses the wasm→JS boundary at most once.
 */
export declare function decodeSlot(
  ctx: MemoryContext,
  dict: Dictionary,
  slot: number,
): string;

/** Decode every slot of `dict` into a `string[]` (memoized per slot). */
export declare function decodeDictionary(ctx: MemoryContext, dict: Dictionary): string[];

/** Decode-cache accounting for `dict` (`{hits:0,misses:0}` if never decoded). */
export declare function decodeStats(dict: Dictionary): { hits: number; misses: number };

/**
 * Unify two dictionaries into one merged unique list plus per-slot index remaps
 * (JS-side; the wasm `unify_dict` kernel is Phase 2, ABI §9). Value-preserving:
 * `merged[remapA[i]] === decode(dictA, i)`.
 */
export declare function unifyDictionaries(
  ctx: MemoryContext,
  dictA: Dictionary,
  dictB: Dictionary,
): DictUnifyResult;

/** Free a dictionary's buffers and drop them from the view registry. */
export declare function freeDictionary(ctx: MemoryContext, dict: Dictionary): void;
