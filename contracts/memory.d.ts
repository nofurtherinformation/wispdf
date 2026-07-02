/**
 * contracts/memory.d.ts — Memory core interface (Phase 1)
 *
 * STATUS: v0.9-draft. Written by task P1.1 (arena allocator, dual wasm builds,
 * loader, viewOf layer). Task P1.2 (columns, dict store, dtype registry,
 * zero-copy slice) FINALIZES this contract — expect additions (Column/Series
 * buffer descriptors, dictionary store, dtype registry) and possible renames
 * before it is marked v1.
 *
 * This is the typed surface the rest of the library builds on. It mirrors the
 * runtime in `src/memory/` and encodes the ABI guarantees from
 * `contracts/wasm-abi.md` (§2 memory ownership + generation counter, §3
 * allocator exports, §9 export list) and the ADRs (ADR-001 single viewOf,
 * ADR-004 dual feature-detected builds).
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
 */
export type ViewDType = 'f64' | 'f32' | 'i32' | 'u32' | 'u8' | 'bool';

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
  | Uint8Array;

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
