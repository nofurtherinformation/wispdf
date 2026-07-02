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

export { createMemoryContext, type MemoryContext } from './context.js';

export {
  DTYPES,
  dtypeInfo,
  type DType,
  type DTypeInfo,
  type TypedArrayCtor,
} from './dtype.js';

export {
  validityBytes,
  getBit,
  setBit,
  clearBit,
} from './bitmap.js';

export {
  createColumn,
  columnToArray,
  sliceColumn,
  freeColumn,
  type Column,
  type ColumnInput,
  type Cell,
} from './column.js';

export {
  writeDictionary,
  decodeSlot,
  decodeDictionary,
  decodeStats,
  unifyDictionaries,
  freeDictionary,
  type Dictionary,
  type DictUnifyResult,
} from './dictionary.js';
