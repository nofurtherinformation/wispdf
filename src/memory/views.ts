/**
 * The single sanctioned `viewOf()` accessor (ADR-001, ABI §2).
 *
 * `memory.grow` detaches every `TypedArray` over the old `ArrayBuffer`. To stay
 * correct without copying, ALL view access goes through one accessor that:
 *   1. caches `(generation, view)` per registered column buffer,
 *   2. checks `mem_generation()` before every use, and
 *   3. on a mismatch, rebuilds ALL registered views over the current
 *      `memory.buffer`.
 *
 * No raw `TypedArray` may be cached anywhere else in the library.
 */

import type { WasmMemoryModule } from './loader.js';

/** Storage dtypes whose data buffer maps to a numeric `TypedArray` (dtypes.md §1). `i64` maps to `BigInt64Array`. */
export type ViewDType = 'f64' | 'f32' | 'i32' | 'u32' | 'u8' | 'bool' | 'i64';

/** Location + shape of a column buffer inside linear memory. */
export interface ColumnBuffer {
  /** Byte offset into `memory.buffer` (16-byte aligned; ABI §3). */
  readonly ptr: number;
  /** Element count (NOT bytes; ABI §4.3). */
  readonly length: number;
  /** Determines the `TypedArray` kind; `bool` is `u8` storage (dtypes.md §1). */
  readonly dtype: ViewDType;
}

/** The concrete `TypedArray` kinds a column view can be. `BigInt64Array` is used for `i64` columns. */
export type ColumnView =
  | Float64Array
  | Float32Array
  | Int32Array
  | Uint32Array
  | Uint8Array
  | BigInt64Array;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyTypedArrayCtor = new (buffer: ArrayBufferLike, byteOffset: number, length: number) => any;

function ctorFor(dtype: ViewDType): AnyTypedArrayCtor {
  switch (dtype) {
    case 'f64':
      return Float64Array;
    case 'f32':
      return Float32Array;
    case 'i32':
      return Int32Array;
    case 'u32':
      return Uint32Array;
    case 'i64':
      return BigInt64Array;
    case 'u8':
    case 'bool':
      return Uint8Array;
  }
}

function keyOf(col: ColumnBuffer): string {
  return `${col.dtype}:${col.ptr}:${col.length}`;
}

/** The `viewOf` accessor returned by {@link createViewOf}. */
export interface ViewOf {
  /**
   * Return a live `TypedArray` for `col` over the current `memory.buffer`,
   * rebuilding every registered view first if memory has grown since last use.
   * The returned view must not be cached by callers — call `viewOf` again.
   */
  (col: ColumnBuffer): ColumnView;
  /** The generation the cache is currently synced to. */
  generation(): number;
  /** Stop tracking `col` (drops it from the registry and cache). */
  forget(col: ColumnBuffer): void;
  /** Drop all tracked columns and cached views. */
  clear(): void;
}

/**
 * Build the memory context's single `viewOf` accessor over `mod`'s linear
 * memory and generation counter.
 */
export function createViewOf(
  mod: Pick<WasmMemoryModule, 'memory' | 'mem_generation'>,
): ViewOf {
  let syncedGeneration = -1;
  let syncedBuffer: ArrayBufferLike | null = null;
  const registry = new Map<string, ColumnBuffer>();
  const cache = new Map<string, ColumnView>();

  function build(col: ColumnBuffer): ColumnView {
    const Ctor = ctorFor(col.dtype);
    return new Ctor(mod.memory.buffer, col.ptr, col.length);
  }

  /** Rebuild all views if the generation advanced or the buffer was replaced. */
  function sync(): void {
    const g = mod.mem_generation();
    if (g !== syncedGeneration || syncedBuffer !== mod.memory.buffer) {
      syncedGeneration = g;
      syncedBuffer = mod.memory.buffer;
      cache.clear();
      for (const [k, col] of registry) {
        cache.set(k, build(col));
      }
    }
  }

  const viewOf = ((col: ColumnBuffer): ColumnView => {
    const k = keyOf(col);
    if (!registry.has(k)) {
      registry.set(k, col);
    }
    sync();
    let view = cache.get(k);
    if (view === undefined) {
      view = build(col);
      cache.set(k, view);
    }
    return view;
  }) as ViewOf;

  viewOf.generation = () => syncedGeneration;
  viewOf.forget = (col: ColumnBuffer) => {
    const k = keyOf(col);
    registry.delete(k);
    cache.delete(k);
  };
  viewOf.clear = () => {
    registry.clear();
    cache.clear();
    syncedGeneration = -1;
    syncedBuffer = null;
  };

  return viewOf;
}
