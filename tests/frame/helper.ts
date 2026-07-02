/**
 * Test harness for the frame layer. Instantiates a wasm build directly (so the full
 * Phase-2 kernel surface is visible, like the kernel conformance tests) and wraps it in
 * a {@link DfRuntime}. `makeDF` builds a DataFrame bound to that runtime.
 */

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { runtimeFromExports, type DfRuntime } from '../../src/frame/runtime.js';
import { DataFrame, type FrameOptions } from '../../src/frame/dataframe.js';
import type { ColumnInput, Cell } from '../../src/memory/column.js';
import type { DType } from '../../src/memory/dtype.js';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIST = join(__dir, '..', '..', 'wasm', 'dist');

/**
 * Default build to use when no explicit `simd` flag is passed.
 * Set `WASM_BUILD=simd` (env var) to exercise the SIMD path (ADR-004).
 */
export const BUILD_SIMD = process.env['WASM_BUILD'] === 'simd';

/** Load a fresh runtime for a test build. Defaults to WASM_BUILD env var (scalar if unset). */
export async function loadRuntimeForTest(simd = BUILD_SIMD): Promise<DfRuntime> {
  const bytes = await readFile(join(WASM_DIST, simd ? 'simd.wasm' : 'scalar.wasm'));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return runtimeFromExports(instance.exports, simd);
}

/** Build a DataFrame from column arrays bound to `rt`. */
export function makeDF(
  rt: DfRuntime,
  cols: Record<string, ColumnInput>,
  dtypes?: Record<string, DType>,
): DataFrame {
  const opts: FrameOptions = dtypes ? { runtime: rt, dtypes } : { runtime: rt };
  return DataFrame.fromColumns(cols, opts);
}

export type { Cell };
