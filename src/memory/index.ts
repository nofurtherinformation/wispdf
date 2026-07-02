/**
 * Memory core (Phase 1): wasm loader + arena allocator handle + the single
 * `viewOf()` accessor. See `contracts/memory.d.ts` for the drafted contract.
 */

export {
  loadWasmModule,
  detectSimd,
  type LoadOptions,
  type WasmExports,
  type WasmMemoryModule,
} from './loader.js';

export {
  createViewOf,
  type ViewOf,
  type ViewDType,
  type ColumnBuffer,
  type ColumnView,
} from './views.js';
