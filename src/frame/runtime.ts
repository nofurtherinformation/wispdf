/**
 * DataFrame runtime: the wasm instance (allocator + memory + full kernel surface)
 * bundled with its MemoryContext. Instantiates the binary directly (loadWasmModule only
 * exposes the memory core) so the frame layer can reach every Phase-2 kernel by ABI name.
 * `init()` installs a process-wide default so `DataFrame.fromColumns({...})` needs no plumbing.
 */

import { detectSimd, type LoadOptions } from '../memory/loader.js';
import { createMemoryContext, type MemoryContext } from '../memory/context.js';
import type { WasmMemoryModule } from '../memory/loader.js';
import type { KernelWasm } from '../expr/index.js';
import type { HashExports } from '../kernels/hash/index.js';
import { FrameError } from './errors.js';

export type FrameWasm = KernelWasm & HashExports;

export type RawKernelFn = (...args: number[]) => number;

export function rawKernel(wasm: FrameWasm, name: string): RawKernelFn {
  const fn = (wasm as unknown as Record<string, RawKernelFn | undefined>)[name];
  if (typeof fn !== 'function') throw new FrameError(`kernel export not found: ${name}`);
  return fn;
}

export interface DfRuntime {

  readonly ctx: MemoryContext;

  readonly wasm: FrameWasm;
}

const isNode =
  typeof process !== 'undefined' && process.versions != null && process.versions.node != null;

async function loadBytes(fileName: string, wasmDir: string | URL | undefined): Promise<BufferSource> {
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
    return readFile(filePath);
  }
  const base = wasmDir ?? new URL('.', import.meta.url);
  const url = new URL(fileName, base);
  const resp = await fetch(url);
  if (!resp.ok) throw new FrameError(`failed to fetch wasm at ${url.toString()}: ${resp.status}`);
  return resp.arrayBuffer();
}

export function runtimeFromExports(exports: WebAssembly.Exports, simd: boolean): DfRuntime {
  const wasm = exports as unknown as FrameWasm;
  const mod: WasmMemoryModule = {
    memory: wasm.memory,
    alloc: wasm.alloc,
    free: wasm.free,
    realloc: wasm.realloc,
    mem_generation: wasm.mem_generation,
    simd,
  };
  return { ctx: createMemoryContext(mod), wasm };
}

export async function loadRuntime(opts: LoadOptions = {}): Promise<DfRuntime> {
  const simd = opts.simd ?? detectSimd();
  const bytes = await loadBytes(simd ? 'simd.wasm' : 'scalar.wasm', opts.wasmDir);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return runtimeFromExports(instance.exports, simd);
}

let defaultRt: DfRuntime | null = null;

export async function init(opts?: LoadOptions): Promise<DfRuntime> {
  defaultRt = await loadRuntime(opts);
  return defaultRt;
}

export function useRuntime(rt: DfRuntime): void {
  defaultRt = rt;
}

export function defaultRuntime(): DfRuntime {
  if (defaultRt === null) {
    throw new FrameError(
      'no DataFrame runtime is loaded — call `await init()` first (or pass a runtime).',
    );
  }
  return defaultRt;
}
