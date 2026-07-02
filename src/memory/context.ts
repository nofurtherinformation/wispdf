/**
 * A `MemoryContext` bundles the loaded wasm module (allocator + linear memory)
 * with its single {@link ViewOf} accessor (ADR-001). Every column / dictionary
 * operation in this layer takes a context: it allocates through `ctx.mod` and
 * reaches bytes only through `ctx.viewOf` — the one sanctioned TypedArray path.
 *
 * There is exactly **one** `viewOf` per module (one linear memory, one generation
 * counter), so there is exactly one context per module.
 */

import { createViewOf, type ViewOf } from './views.js';
import type { WasmMemoryModule } from './loader.js';

/** The allocator + the single `viewOf` accessor over one module's linear memory. */
export interface MemoryContext {
  /** The loaded memory-core module (`alloc`/`free`/`realloc`/`memory`, ABI §3/§9). */
  readonly mod: WasmMemoryModule;
  /** The one sanctioned `TypedArray` accessor over `mod.memory` (ADR-001). */
  readonly viewOf: ViewOf;
}

/** Build the single {@link MemoryContext} for a loaded module. */
export function createMemoryContext(mod: WasmMemoryModule): MemoryContext {
  return { mod, viewOf: createViewOf(mod) };
}
