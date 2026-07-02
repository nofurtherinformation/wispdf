/**
 * Test harness for the expression layer.
 *
 * Instantiates a wasm build **directly** (like the kernel conformance tests) so the
 * full Phase-2 kernel surface is visible, then wraps it in a `MemoryContext` +
 * `FrameView` the compiler consumes. `makeFrame` builds columns from plain JS arrays
 * via the Phase-1 memory layer.
 */

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { createMemoryContext, type MemoryContext } from '../../src/memory/context.js';
import type { WasmMemoryModule } from '../../src/memory/loader.js';
import {
  createColumn,
  columnToArray,
  freeColumn,
  type Column,
  type Cell,
  type ColumnInput,
} from '../../src/memory/column.js';
import type { DType } from '../../src/memory/dtype.js';
import type { FrameView, KernelWasm } from '../../src/expr/frame.js';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIST = join(__dir, '..', '..', 'wasm', 'dist');

/**
 * Default build to use when no explicit `simd` flag is passed.
 * Set `WASM_BUILD=simd` (env var) to exercise the SIMD path (ADR-004).
 */
export const BUILD_SIMD = process.env['WASM_BUILD'] === 'simd';

export interface TestEnv {
  ctx: MemoryContext;
  wasm: KernelWasm;
}

/** Load a wasm build and expose both the memory context and the full kernel exports. */
export async function loadEnv(simd = BUILD_SIMD): Promise<TestEnv> {
  const bytes = await readFile(join(WASM_DIST, simd ? 'simd.wasm' : 'scalar.wasm'));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const ex = instance.exports as unknown as KernelWasm & WasmMemoryModule;
  const mod: WasmMemoryModule = {
    memory: ex.memory,
    alloc: ex.alloc,
    free: ex.free,
    realloc: ex.realloc,
    mem_generation: ex.mem_generation,
    simd,
  };
  return { ctx: createMemoryContext(mod), wasm: ex };
}

/** A column spec: a dtype + the JS values (may include `null`, `NaN`, strings, booleans). */
export interface ColSpec {
  dtype: DType;
  values: ColumnInput;
}

/** A test frame that owns its source columns and implements {@link FrameView}. */
export class TestFrame implements FrameView {
  readonly length: number;
  readonly ctx: MemoryContext;
  readonly wasm: KernelWasm;
  private readonly cols = new Map<string, Column>();
  private readonly dtypes = new Map<string, DType>();

  constructor(env: TestEnv, specs: Record<string, ColSpec>) {
    this.ctx = env.ctx;
    this.wasm = env.wasm;
    let len = -1;
    for (const [name, spec] of Object.entries(specs)) {
      const col = createColumn(env.ctx, spec.dtype, spec.values);
      if (len === -1) len = col.length;
      else if (col.length !== len) throw new Error(`column '${name}' length ${col.length} != ${len}`);
      this.cols.set(name, col);
      this.dtypes.set(name, spec.dtype);
    }
    this.length = len === -1 ? 0 : len;
  }

  getColumn(name: string): Column | undefined {
    return this.cols.get(name);
  }
  dtypeOf(name: string): DType | undefined {
    return this.dtypes.get(name);
  }
  columnNames(): readonly string[] {
    return [...this.cols.keys()];
  }

  /** Free every source column (call in teardown). */
  free(): void {
    for (const col of this.cols.values()) freeColumn(this.ctx, col);
    this.cols.clear();
  }
}

/** Read a result column to a JS array and free it. */
export function takeColumn(ctx: MemoryContext, col: Column): Cell[] {
  const out = columnToArray(ctx, col);
  freeColumn(ctx, col);
  return out;
}
