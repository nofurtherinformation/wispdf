/**
 * The surface the expression compiler consumes from a frame (Phase 3, P3.1).
 *
 * The DataFrame layer (P3.2) implements {@link FrameView} and hands it to
 * {@link compile}/{@link compileFilter}. It is deliberately tiny: column lookup by
 * name, a row count, and the memory + kernel handles the compiler needs to allocate
 * temporaries and dispatch kernels. Keeping it small keeps the compiler decoupled
 * from the concrete DataFrame implementation.
 */

import type { MemoryContext } from '../memory/context.js';
import type { Column } from '../memory/column.js';
import type { Schema } from './dtypes.js';

/**
 * The loaded wasm instance's kernel exports. Every Phase-2 kernel (ABI §9) lives on
 * the **same** instance as the Phase-1 memory core, so `memory`/`alloc`/`free` here
 * are the identical bindings behind {@link MemoryContext.mod} — allocations made
 * through the context are visible to these kernels (same linear memory).
 *
 * Kernel functions are reached by their ABI name; {@link callKernel} does the typed
 * lookup so the rest of the compiler never touches the index signature directly.
 */
export interface KernelWasm {
  readonly memory: WebAssembly.Memory;
  alloc(size: number): number;
  free(ptr: number): void;
  realloc(ptr: number, newSize: number): number;
  mem_generation(): number;
}

/** A kernel export: flat C ABI, wasm value types in/out (ABI §5). */
export type KernelFn = (...args: number[]) => number;

/** Look up and invoke kernel `name` on `wasm` (ABI §6 naming). */
export function callKernel(wasm: KernelWasm, name: string, args: readonly number[]): number {
  const fn = (wasm as unknown as Record<string, KernelFn | undefined>)[name];
  if (typeof fn !== 'function') {
    throw new Error(`kernel export not found: ${name}`);
  }
  return fn(...args);
}

/**
 * Like {@link callKernel} but accepts `bigint` args and may return `bigint` — needed
 * for i64 scalar kernels (e.g. `add_i64_scalar`) and i64 reductions (e.g. `sum_i64_null`).
 * The WebAssembly JS-API automatically maps wasm `i64` ↔ JS `BigInt`.
 */
export function callKernelBigInt(
  wasm: KernelWasm,
  name: string,
  args: readonly (number | bigint)[],
): number | bigint {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fn = (wasm as unknown as Record<string, ((...a: any[]) => number | bigint) | undefined>)[name];
  if (typeof fn !== 'function') {
    throw new Error(`kernel export not found: ${name}`);
  }
  return fn(...args);
}

/**
 * Everything the compiler needs from a frame to type-check, allocate, and run a plan.
 * Implements {@link Schema} (via `dtypeOf`/`columnNames`) so it doubles as the type
 * checker's schema.
 */
export interface FrameView extends Schema {
  /** Row count (`size`, dtypes.md §4.4). Every column has this length. */
  readonly length: number;
  /** Allocator + the single `viewOf` accessor over the wasm linear memory. */
  readonly ctx: MemoryContext;
  /** The kernel exports (same instance as `ctx.mod`, see {@link KernelWasm}). */
  readonly wasm: KernelWasm;
  /** Resolve column `name` to its {@link Column}, or `undefined` if absent. */
  getColumn(name: string): Column | undefined;
}
