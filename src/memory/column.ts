/**
 * Column representation (Phase 1, deliverable P1.2 §1/§4; ABI §4).
 *
 * A {@link Column} is a small JS descriptor over buffers living in wasm linear
 * memory (ADR-001); JS never owns the bytes. Per ABI §4 it carries `dtype`,
 * `length`, a `dataPtr`, a `validityPtr` (`0` = all-valid), and — for `utf8` —
 * the shared {@link Dictionary}.
 *
 * ## Zero-copy slice & the offset convention (deliverable §4)
 * `sliceColumn` shares the parent's buffers; no bytes move. Two offset kinds:
 *
 *  - **data** is byte-addressable, so a slice bakes its start into `dataPtr`:
 *    `dataPtr = parent.dataPtr + start * dtype.size`. Element `i`'s data byte
 *    offset is `dataPtr + i * size`. **`dataPtr` is exactly the base pointer a
 *    kernel receives** — the byte-offset math happens here, in the memory layer,
 *    so Phase-2 kernels take `(dataPtr, len)` unchanged. Alignment is preserved:
 *    offsetting by whole elements keeps the natural alignment `alloc` guarantees.
 *
 *  - **validity** is *bit*-addressed, so a slice cannot move the pointer; it keeps
 *    the parent's `validityPtr` and records `validityBitOffset` (the bit index of
 *    element 0). Element `i`'s validity bit is `validityBitOffset + i` in the
 *    bitmap at `validityPtr`. For a root column `validityBitOffset == 0`. A
 *    Phase-2 kernel consuming a sliced column's validity must honor this bit
 *    offset (or the memory layer realigns first); root columns — the common case —
 *    pass `validityPtr` directly. `validityPtr == 0` always means all-valid.
 */

import { DTYPES, type DType } from './dtype.js';
import { validityBytes, getBit, setBit } from './bitmap.js';
import type { MemoryContext } from './context.js';
import {
  writeDictionary,
  decodeSlot,
  freeDictionary,
  type Dictionary,
} from './dictionary.js';

/** A column descriptor over wasm buffers (see module doc; ABI §4). */
export interface Column {
  /** Storage dtype (`contracts/dtypes.md` §1). */
  readonly dtype: DType;
  /** Element count of this (possibly sliced) column. */
  readonly length: number;
  /** Base byte offset of element 0's data (slice start already baked in). */
  readonly dataPtr: number;
  /** Validity bitmap pointer, or `0` for an all-valid column (ABI §4.1). */
  readonly validityPtr: number;
  /** Bit index of element 0 within the bitmap at `validityPtr` (0 for roots). */
  readonly validityBitOffset: number;
  /** The shared dictionary for a `utf8` column; `null` for every other dtype. */
  readonly dict: Dictionary | null;
  /** True if this column owns its buffers (a root); slices share and own nothing. */
  readonly owned: boolean;
}

/** JS value shapes accepted by {@link createColumn}, per dtype. */
export type ColumnInput =
  | ArrayLike<number | null | undefined>
  | ArrayLike<boolean | null | undefined>
  | ArrayLike<string | null | undefined>;

/** One decoded column cell: a value, or `null` for a null slot. */
export type Cell = number | boolean | string | null;

/** Structural handle for the single bulk-write method shared by all TypedArrays. */
interface Settable {
  set(src: ArrayLike<number>, offset?: number): void;
}

function isNullish(v: unknown): boolean {
  return v === null || v === undefined;
}

function root(
  dtype: DType,
  length: number,
  dataPtr: number,
  validityPtr: number,
  dict: Dictionary | null,
): Column {
  return { dtype, length, dataPtr, validityPtr, validityBitOffset: 0, dict, owned: true };
}

/**
 * Build a column from JS `values` for `dtype`. Two paths:
 *   - **fast path** — a matching `TypedArray` (no nulls possible): bulk-copy into
 *     linear memory, `validityPtr = 0`;
 *   - **slow path** — a plain array: detect `null`/`undefined` (only these are
 *     nulls — a `NaN` is a *value*, dtypes.md §4) and build the validity bitmap.
 */
export function createColumn(ctx: MemoryContext, dtype: DType, values: ColumnInput): Column {
  if (dtype === 'utf8') {
    return createStringColumn(ctx, values as ArrayLike<string | null | undefined>);
  }
  if (dtype === 'bool') {
    return createBoolColumn(ctx, values as ArrayLike<boolean | null | undefined>);
  }
  return createNumericColumn(ctx, dtype, values as ArrayLike<number | null | undefined>);
}

function numericFastPath(dtype: DType, values: ArrayLike<unknown>): boolean {
  switch (dtype) {
    case 'f64':
      return values instanceof Float64Array;
    case 'f32':
      return values instanceof Float32Array;
    case 'i32':
      return values instanceof Int32Array;
    case 'u32':
      return values instanceof Uint32Array;
    default:
      return false;
  }
}

function createNumericColumn(
  ctx: MemoryContext,
  dtype: DType,
  values: ArrayLike<number | null | undefined>,
): Column {
  const info = DTYPES[dtype];
  const len = values.length;

  if (numericFastPath(dtype, values)) {
    const dataPtr = ctx.mod.alloc(len * info.size);
    const view = ctx.viewOf({ ptr: dataPtr, length: len, dtype: info.view }) as unknown as Settable;
    view.set(values as unknown as ArrayLike<number>);
    return root(dtype, len, dataPtr, 0, null);
  }

  // Slow path: scan for nulls first (JS only), then allocate all buffers, then write.
  let nulls = 0;
  for (let i = 0; i < len; i++) if (isNullish(values[i])) nulls++;

  const dataPtr = ctx.mod.alloc(len * info.size);
  const validityPtr = nulls > 0 ? ctx.mod.alloc(validityBytes(len)) : 0;

  const view = ctx.viewOf({ ptr: dataPtr, length: len, dtype: info.view });
  let vbits: Uint8Array | null = null;
  if (validityPtr !== 0) {
    vbits = ctx.viewOf({ ptr: validityPtr, length: validityBytes(len), dtype: 'u8' }) as Uint8Array;
    vbits.fill(0); // clear pad bits + start all-null, then set valid bits
  }
  for (let i = 0; i < len; i++) {
    const v = values[i];
    if (isNullish(v)) {
      view[i] = 0;
    } else {
      view[i] = v as number; // TypedArray assignment coerces (f32 rounds, int truncates)
      if (vbits) setBit(vbits, i);
    }
  }
  return root(dtype, len, dataPtr, validityPtr, null);
}

function createBoolColumn(
  ctx: MemoryContext,
  values: ArrayLike<boolean | null | undefined>,
): Column {
  const len = values.length;

  if (values instanceof Uint8Array) {
    // Fast path: caller-provided 0/1 storage, no nulls.
    const dataPtr = ctx.mod.alloc(len);
    const view = ctx.viewOf({ ptr: dataPtr, length: len, dtype: 'bool' }) as unknown as Settable;
    view.set(values);
    return root('bool', len, dataPtr, 0, null);
  }

  let nulls = 0;
  for (let i = 0; i < len; i++) if (isNullish(values[i])) nulls++;

  const dataPtr = ctx.mod.alloc(len);
  const validityPtr = nulls > 0 ? ctx.mod.alloc(validityBytes(len)) : 0;

  const view = ctx.viewOf({ ptr: dataPtr, length: len, dtype: 'bool' }) as Uint8Array;
  let vbits: Uint8Array | null = null;
  if (validityPtr !== 0) {
    vbits = ctx.viewOf({ ptr: validityPtr, length: validityBytes(len), dtype: 'u8' }) as Uint8Array;
    vbits.fill(0);
  }
  for (let i = 0; i < len; i++) {
    const v = values[i];
    if (isNullish(v)) {
      view[i] = 0;
    } else {
      view[i] = v ? 1 : 0;
      if (vbits) setBit(vbits, i);
    }
  }
  return root('bool', len, dataPtr, validityPtr, null);
}

function createStringColumn(
  ctx: MemoryContext,
  values: ArrayLike<string | null | undefined>,
): Column {
  const len = values.length;
  const index = new Map<string, number>();
  const uniques: string[] = [];
  const idxData = new Int32Array(len); // JS staging copy of the i32 indices
  const nullMask = new Uint8Array(len); // 1 = null row
  let nulls = 0;

  for (let i = 0; i < len; i++) {
    const v = values[i];
    if (isNullish(v)) {
      nulls++;
      nullMask[i] = 1;
      idxData[i] = 0;
    } else {
      const s = v as string;
      let j = index.get(s);
      if (j === undefined) {
        j = uniques.length;
        uniques.push(s);
        index.set(s, j);
      }
      idxData[i] = j;
    }
  }

  // Allocate the index + validity buffers, then build the dictionary (which
  // allocs its own buffers). Take views only after ALL allocations complete.
  const dataPtr = ctx.mod.alloc(len * 4);
  const validityPtr = nulls > 0 ? ctx.mod.alloc(validityBytes(len)) : 0;
  const dict = writeDictionary(ctx, uniques);

  const idxView = ctx.viewOf({ ptr: dataPtr, length: len, dtype: 'i32' }) as unknown as Settable;
  idxView.set(idxData);
  if (validityPtr !== 0) {
    const vbits = ctx.viewOf({ ptr: validityPtr, length: validityBytes(len), dtype: 'u8' }) as Uint8Array;
    vbits.fill(0);
    for (let i = 0; i < len; i++) if (nullMask[i] === 0) setBit(vbits, i);
  }
  return root('utf8', len, dataPtr, validityPtr, dict);
}

/**
 * Export a column back to a JS array, with null slots as `null` (deliverable §1).
 * `f64`/`f32` `NaN`s round-trip as values; `bool` yields `boolean`; `utf8` yields
 * memoized-decoded strings.
 */
export function columnToArray(ctx: MemoryContext, col: Column): Cell[] {
  const { length, validityPtr, validityBitOffset } = col;
  const validity =
    validityPtr === 0
      ? null
      : (ctx.viewOf({
          ptr: validityPtr,
          length: validityBytes(validityBitOffset + length),
          dtype: 'u8',
        }) as Uint8Array);

  const out = new Array<Cell>(length);

  if (col.dtype === 'utf8') {
    const idx = ctx.viewOf({ ptr: col.dataPtr, length, dtype: 'i32' }) as Int32Array;
    const dict = col.dict!;
    for (let i = 0; i < length; i++) {
      out[i] =
        validity && !getBit(validity, validityBitOffset + i)
          ? null
          : decodeSlot(ctx, dict, idx[i]!);
    }
    return out;
  }

  const info = DTYPES[col.dtype];
  const data = ctx.viewOf({ ptr: col.dataPtr, length, dtype: info.view });
  const isBool = col.dtype === 'bool';
  for (let i = 0; i < length; i++) {
    if (validity && !getBit(validity, validityBitOffset + i)) {
      out[i] = null;
    } else {
      out[i] = isBool ? data[i] !== 0 : data[i]!;
    }
  }
  return out;
}

/**
 * Zero-copy slice `[start, end)` sharing the parent's buffers (deliverable §4).
 * `start`/`end` are clamped to `[0, length]`; `end < start` yields an empty slice.
 * See the module doc for the data/validity offset convention. Slice-of-slice
 * composes: offsets accumulate, buffers stay shared.
 */
export function sliceColumn(col: Column, start: number, end: number): Column {
  const s = Math.max(0, Math.min(start, col.length));
  const e = Math.max(s, Math.min(end, col.length));
  const n = e - s;
  const size = DTYPES[col.dtype].size;
  return {
    dtype: col.dtype,
    length: n,
    dataPtr: col.dataPtr + s * size,
    validityPtr: col.validityPtr,
    validityBitOffset: col.validityBitOffset + s,
    dict: col.dict,
    owned: false, // a slice never owns the shared buffers
  };
}

/**
 * Free a column's owned buffers and drop them from the view registry. A no-op for
 * slices (`owned === false`) — they share buffers owned by the root. Frees the
 * dictionary too for an owned `utf8` column.
 */
export function freeColumn(ctx: MemoryContext, col: Column): void {
  if (!col.owned) return;
  const info = DTYPES[col.dtype];
  ctx.viewOf.forget({ ptr: col.dataPtr, length: col.length, dtype: info.view });
  ctx.mod.free(col.dataPtr);
  if (col.validityPtr !== 0) {
    ctx.viewOf.forget({
      ptr: col.validityPtr,
      length: validityBytes(col.validityBitOffset + col.length),
      dtype: 'u8',
    });
    ctx.mod.free(col.validityPtr);
  }
  if (col.dict !== null) {
    freeDictionary(ctx, col.dict);
  }
}
