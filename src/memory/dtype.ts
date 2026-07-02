/**
 * Schema / dtype registry (Phase 1, deliverable P1.2 §3).
 *
 * One descriptor per v1 storage dtype (`contracts/dtypes.md` §1). The descriptor
 * carries everything the column, dictionary, and (Phase 2) kernel/expr layers need
 * to lay a dtype out in linear memory and name its kernels:
 *   - `size`   — bytes per stored element (utf8 stores an `i32` dictionary index),
 *   - `view`   — the {@link ViewDType} its data buffer maps to through `viewOf`,
 *   - `ctor`   — the matching `TypedArray` constructor for a JS-side staging copy,
 *   - `wasm`   — the kernel-name dtype token (ABI §6 `[family_]op_dtype[_variant]`),
 *   - `float`  — whether `NaN`/`±inf` are *values* (floats), never nulls (dtypes.md §4).
 */

import type { ViewDType } from './views.js';

/** The v1 column dtypes (`contracts/dtypes.md` §1). `utf8` is dict-encoded (ABI §4.4). */
export type DType = 'f64' | 'f32' | 'i32' | 'u32' | 'bool' | 'utf8';

/** `TypedArray` constructors a column data / auxiliary buffer can map to. */
export type TypedArrayCtor =
  | Float64ArrayConstructor
  | Float32ArrayConstructor
  | Int32ArrayConstructor
  | Uint32ArrayConstructor
  | Uint8ArrayConstructor;

/** Static description of one v1 dtype (see module doc). */
export interface DTypeInfo {
  /** The dtype this describes. */
  readonly name: DType;
  /** Bytes per stored element. `utf8` = 4 (the `i32` index into its dictionary). */
  readonly size: number;
  /** `viewOf` dtype for this column's *data* buffer (`utf8` → its `i32` indices). */
  readonly view: ViewDType;
  /** `TypedArray` constructor matching {@link view} (for staging copies). */
  readonly ctor: TypedArrayCtor;
  /** Kernel-name dtype token (ABI §6). Same as {@link name} for every v1 dtype. */
  readonly wasm: string;
  /** True for `f64`/`f32`: a `NaN`/`±inf` is a valid *value*, never a null (dtypes.md §4). */
  readonly float: boolean;
}

/** The dtype registry: `DTYPES[dtype]` → its {@link DTypeInfo}. */
export const DTYPES: Record<DType, DTypeInfo> = {
  f64: { name: 'f64', size: 8, view: 'f64', ctor: Float64Array, wasm: 'f64', float: true },
  f32: { name: 'f32', size: 4, view: 'f32', ctor: Float32Array, wasm: 'f32', float: true },
  i32: { name: 'i32', size: 4, view: 'i32', ctor: Int32Array, wasm: 'i32', float: false },
  u32: { name: 'u32', size: 4, view: 'u32', ctor: Uint32Array, wasm: 'u32', float: false },
  // `bool` value storage is one u8 per element (0/1); validity is a separate bitmap.
  bool: { name: 'bool', size: 1, view: 'bool', ctor: Uint8Array, wasm: 'bool', float: false },
  // `utf8` data buffer is `i32[len]` dictionary indices (ABI §4.4); dict is separate.
  utf8: { name: 'utf8', size: 4, view: 'i32', ctor: Int32Array, wasm: 'utf8', float: false },
};

/** Descriptor for `dtype`. Throws on an unknown dtype (helpful for API callers). */
export function dtypeInfo(dtype: DType): DTypeInfo {
  const info = DTYPES[dtype];
  if (info === undefined) {
    throw new Error(`unknown dtype: ${String(dtype)}`);
  }
  return info;
}
