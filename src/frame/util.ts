/** Shared frame helpers: kernel dtype tokens, bit-0 validity realignment for sliced
 * columns (ABI §4.1 has no bit-offset param), and verbatim dictionary copying. */

import type { MemoryContext } from '../memory/context.js';
import { DTYPES, type DType } from '../memory/dtype.js';
import { validityBytes, getBit, setBit } from '../memory/bitmap.js';
import type { Column } from '../memory/column.js';
import type { Dictionary } from '../memory/dictionary.js';

/**
 * Kernel dtype token for sort/gather/filter.
 * Temporal types route to their physical kernel token (ADR-010).
 */
export function storageToken(dtype: DType): string {
  if (dtype === 'bool') return 'u8';
  if (dtype === 'utf8') return 'i32';
  return DTYPES[dtype].wasm; // date32→'i32', timestamp→'i64', others → unchanged
}

export interface AlignedValidity {

  readonly ptr: number;

  readonly owns: boolean;
}

export function alignValidity(ctx: MemoryContext, col: Column): AlignedValidity {
  if (col.validityPtr === 0) return { ptr: 0, owns: false };
  const bitOff = col.validityBitOffset;
  if ((bitOff & 7) === 0) return { ptr: col.validityPtr + (bitOff >> 3), owns: false };
  const len = col.length;
  const bytes = validityBytes(len);
  const outPtr = ctx.mod.alloc(Math.max(bytes, 1));
  const src = ctx.viewOf({
    ptr: col.validityPtr,
    length: validityBytes(bitOff + len),
    dtype: 'u8',
  }) as Uint8Array;
  const dst = ctx.viewOf({ ptr: outPtr, length: bytes, dtype: 'u8' }) as Uint8Array;
  dst.fill(0);
  for (let i = 0; i < len; i++) if (getBit(src, bitOff + i)) setBit(dst, i);
  return { ptr: outPtr, owns: true };
}

export function freeAligned(ctx: MemoryContext, v: AlignedValidity): void {
  if (v.owns && v.ptr !== 0) {
    ctx.viewOf.forget({ ptr: v.ptr, length: 1, dtype: 'u8' });
    ctx.mod.free(v.ptr);
  }
}

export function copyDictionary(ctx: MemoryContext, dict: Dictionary): Dictionary {
  const offBytes = (dict.count + 1) * 4;
  const offsetsPtr = ctx.mod.alloc(Math.max(offBytes, 1));
  const bytesPtr = ctx.mod.alloc(Math.max(dict.bytesLen, 1));
  (ctx.viewOf({ ptr: offsetsPtr, length: dict.count + 1, dtype: 'i32' }) as Int32Array).set(
    ctx.viewOf({ ptr: dict.offsetsPtr, length: dict.count + 1, dtype: 'i32' }) as Int32Array,
  );
  if (dict.bytesLen > 0) {
    (ctx.viewOf({ ptr: bytesPtr, length: dict.bytesLen, dtype: 'u8' }) as Uint8Array).set(
      ctx.viewOf({ ptr: dict.bytesPtr, length: dict.bytesLen, dtype: 'u8' }) as Uint8Array,
    );
  }
  return { count: dict.count, offsetsPtr, bytesPtr, bytesLen: dict.bytesLen };
}

export function dataBytes(dtype: DType, n: number): number {
  return Math.max(n * DTYPES[dtype].size, 1);
}
