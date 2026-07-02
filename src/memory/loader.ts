/**
 * WASM memory-core loader with SIMD feature detection (ADR-004).
 *
 * Feature-detects SIMD via `WebAssembly.validate` on a tiny inlined SIMD module,
 * then instantiates the matching binary — `simd.wasm` or `scalar.wasm` — over
 * either a Node filesystem path or a browser URL. The module exports the
 * Phase-1 memory core (ABI §9): `memory`, `alloc`, `free`, `realloc`,
 * `mem_generation`.
 *
 * View access is NOT done here — hold no `TypedArray` outside `viewOf`
 * (ADR-001). See {@link createViewOf} in `./views.ts`.
 */

/**
 * Minimal SIMD module for `WebAssembly.validate` feature detection (ADR-004):
 * `(module (func (result v128) i32.const 0 i8x16.splat i8x16.popcnt))`.
 * `validate` returns `true` iff the runtime supports the SIMD128 proposal.
 */
const SIMD_DETECT_MODULE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60, 0x00,
  0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00, 0x41, 0x00,
  0xfd, 0x0f, 0xfd, 0x62, 0x0b,
]);

/** True iff the current runtime supports wasm SIMD128 (ADR-004). */
export function detectSimd(): boolean {
  try {
    return WebAssembly.validate(SIMD_DETECT_MODULE);
  } catch {
    return false;
  }
}

const isNode =
  typeof process !== 'undefined' &&
  process.versions != null &&
  process.versions.node != null;

/** Raw exports of the memory-core wasm module (ABI §9, Phase 1). */
export interface WasmExports {
  /** The module's single linear memory (ADR-001; all column bytes live here). */
  readonly memory: WebAssembly.Memory;
  /** `alloc(size) -> ptr`: 16-byte-aligned, `0` on OOM (ABI §3). */
  alloc(size: number): number;
  /** `free(ptr)`: `free(0)` is a no-op (ABI §3). */
  free(ptr: number): void;
  /** `realloc(ptr, newSize) -> ptr`: preserves contents; `0` on OOM (ABI §3). */
  realloc(ptr: number, newSize: number): number;
  /** `mem_generation()`: changes on every successful `memory.grow` (ABI §2). */
  mem_generation(): number;
}

/** A loaded memory-core module plus which build was selected. */
export interface WasmMemoryModule extends WasmExports {
  /** `true` if the SIMD build was loaded, `false` for the scalar build. */
  readonly simd: boolean;
}

/** Options for {@link loadWasmModule}. */
export interface LoadOptions {
  /** Force a build. Default: auto-detect via {@link detectSimd}. */
  simd?: boolean;
  /**
   * Location of `scalar.wasm` / `simd.wasm`.
   *  - Node: a filesystem directory path, or a `file:` / directory `URL`.
   *  - Browser: a base `URL` (or URL string) the two files are fetched under.
   *
   * Default: resolved relative to this module (the built binaries are copied
   * next to the JS bundle in `dist/`).
   */
  wasmDir?: string | URL;
}

async function loadBytes(
  fileName: string,
  wasmDir: string | URL | undefined,
): Promise<BufferSource> {
  if (isNode) {
    const [{ readFile }, { fileURLToPath }] = await Promise.all([
      import('node:fs/promises'),
      import('node:url'),
    ]);
    let filePath: string;
    if (wasmDir === undefined) {
      filePath = fileURLToPath(new URL(fileName, import.meta.url));
    } else if (wasmDir instanceof URL) {
      filePath = fileURLToPath(new URL(fileName, wasmDir));
    } else if (wasmDir.startsWith('file:')) {
      filePath = fileURLToPath(new URL(fileName, wasmDir));
    } else {
      const { join } = await import('node:path');
      filePath = join(wasmDir, fileName);
    }
    return await readFile(filePath);
  }

  const base = wasmDir ?? new URL('.', import.meta.url);
  const url = new URL(fileName, base);
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to fetch wasm at ${url.toString()}: ${resp.status}`);
  }
  return await resp.arrayBuffer();
}

/**
 * Feature-detect, load, and instantiate the memory-core wasm module.
 * Runs once per page/process (ADR-004); callers cache the returned module.
 */
export async function loadWasmModule(
  opts: LoadOptions = {},
): Promise<WasmMemoryModule> {
  const simd = opts.simd ?? detectSimd();
  const fileName = simd ? 'simd.wasm' : 'scalar.wasm';
  const bytes = await loadBytes(fileName, opts.wasmDir);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const ex = instance.exports as unknown as WasmExports;
  return {
    memory: ex.memory,
    alloc: ex.alloc,
    free: ex.free,
    realloc: ex.realloc,
    mem_generation: ex.mem_generation,
    simd,
  };
}
