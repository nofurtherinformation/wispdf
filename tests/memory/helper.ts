import { loadWasmModule, type WasmMemoryModule } from '../../src/memory/loader.js';
import { createMemoryContext, type MemoryContext } from '../../src/memory/context.js';

/** Directory holding the built scalar.wasm / simd.wasm (produced by build.sh). */
export const WASM_DIR = new URL('../../wasm/dist/', import.meta.url);

/**
 * Default build to use when no explicit `simd` flag is passed.
 * Set `WASM_BUILD=simd` (env var) to exercise the SIMD path (ADR-004).
 */
export const BUILD_SIMD = process.env['WASM_BUILD'] === 'simd';

/** Load a fresh memory-core module instance for a test. */
export function loadForTest(simd = BUILD_SIMD): Promise<WasmMemoryModule> {
  return loadWasmModule({ simd, wasmDir: WASM_DIR });
}

/** Load a fresh module + its single {@link MemoryContext} (allocator + viewOf). */
export async function ctxForTest(simd = BUILD_SIMD): Promise<MemoryContext> {
  return createMemoryContext(await loadForTest(simd));
}

/** Both builds — the allocator is identical in each and both must pass. */
export const BUILDS: ReadonlyArray<{ label: string; simd: boolean }> = [
  { label: 'scalar', simd: false },
  { label: 'simd', simd: true },
];
