import { loadWasmModule, type WasmMemoryModule } from '../../src/memory/loader.js';

/** Directory holding the built scalar.wasm / simd.wasm (produced by build.sh). */
export const WASM_DIR = new URL('../../wasm/dist/', import.meta.url);

/** Load a fresh memory-core module instance for a test. */
export function loadForTest(simd: boolean): Promise<WasmMemoryModule> {
  return loadWasmModule({ simd, wasmDir: WASM_DIR });
}

/** Both builds — the allocator is identical in each and both must pass. */
export const BUILDS: ReadonlyArray<{ label: string; simd: boolean }> = [
  { label: 'scalar', simd: false },
  { label: 'simd', simd: true },
];
