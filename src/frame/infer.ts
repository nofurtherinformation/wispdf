/** dtype inference for fromColumns/fromRecords: typed arrays map unambiguously; a plain
 * array uses its first non-null cell (number→f64, boolean→bool, string→utf8); empty→f64. */

import type { DType } from '../memory/dtype.js';
import type { ColumnInput } from '../memory/column.js';

export function inferDType(input: ColumnInput): DType {
  if (input instanceof BigInt64Array) return 'i64';
  if (input instanceof Float64Array) return 'f64';
  if (input instanceof Float32Array) return 'f32';
  if (input instanceof Int32Array) return 'i32';
  if (input instanceof Uint32Array) return 'u32';
  if (input instanceof Uint8Array) return 'bool';
  const arr = input as ArrayLike<unknown>;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v === null || v === undefined) continue;
    if (typeof v === 'bigint') return 'i64'; // bigint → i64 ONLY (never number)
    if (typeof v === 'number') return 'f64';
    if (typeof v === 'boolean') return 'bool';
    if (typeof v === 'string') return 'utf8';
  }
  return 'f64';
}
