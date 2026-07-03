/**
 * Sort permutation for sortValues (dtypes.md §4.6): the stable `argsort_dt` kernel, multi-key
 * threaded last-key-first (ABI §9 C); nulls sort last both directions. utf8 keys sort
 * lexicographically via a per-row string-rank buffer; bool widens to i32. The v1.2 scratch_ptr
 * amendment is handled by arity detection, so this works pre- and post-amendment.
 */

import type { Column } from '../memory/column.js';
import { DTYPES } from '../memory/dtype.js';
import { decodeDictionary } from '../memory/dictionary.js';
import type { DfRuntime, FrameWasm } from './runtime.js';
import { rawKernel } from './runtime.js';
import { alignValidity, freeAligned } from './util.js';

export interface Perm {
  readonly ptr: number;
  readonly len: number;
}

export function sortPerm(rt: DfRuntime, keyCols: Column[], descending: boolean[]): Perm {
  const { ctx, wasm } = rt;
  const len = keyCols.length > 0 ? keyCols[0]!.length : 0;

  const permPtr = ctx.mod.alloc(Math.max(len * 4, 1));
  const perm = ctx.viewOf({ ptr: permPtr, length: len, dtype: 'i32' }) as Int32Array;
  for (let i = 0; i < len; i++) perm[i] = i;

  const needScratch = rawKernel(wasm, 'argsort_i32').length >= 6;
  const scratchPtr = needScratch ? ctx.mod.alloc(Math.max(len * 4, 1)) : 0;

  try {
    for (let k = keyCols.length - 1; k >= 0; k--) {
      applyKey(rt, keyCols[k]!, descending[k] === true, permPtr, len, scratchPtr, needScratch);
    }
  } finally {
    if (scratchPtr !== 0) {
      ctx.viewOf.forget({ ptr: scratchPtr, length: len, dtype: 'i32' });
      ctx.mod.free(scratchPtr);
    }
  }
  return { ptr: permPtr, len };
}

function callArgsort(
  wasm: FrameWasm,
  name: string,
  dataPtr: number,
  vp: number,
  permPtr: number,
  len: number,
  desc: number,
  scratchPtr: number,
  needScratch: boolean,
): void {
  const fn = rawKernel(wasm, name);
  if (needScratch) fn(dataPtr, vp, permPtr, len, desc, scratchPtr);
  else fn(dataPtr, vp, permPtr, len, desc);
}

function applyKey(
  rt: DfRuntime,
  col: Column,
  descending: boolean,
  permPtr: number,
  len: number,
  scratchPtr: number,
  needScratch: boolean,
): void {
  const { ctx, wasm } = rt;
  const desc = descending ? 1 : 0;
  const av = alignValidity(ctx, col);
  try {
    if (col.dtype === 'utf8') {
      const rankPtr = buildUtf8Rank(rt, col, len);
      try {
        callArgsort(wasm, 'argsort_i32', rankPtr, av.ptr, permPtr, len, desc, scratchPtr, needScratch);
      } finally {
        ctx.viewOf.forget({ ptr: rankPtr, length: len, dtype: 'i32' });
        ctx.mod.free(rankPtr);
      }
    } else if (col.dtype === 'bool') {
      const widePtr = widenBoolToI32(rt, col, len);
      try {
        callArgsort(wasm, 'argsort_i32', widePtr, av.ptr, permPtr, len, desc, scratchPtr, needScratch);
      } finally {
        ctx.viewOf.forget({ ptr: widePtr, length: len, dtype: 'i32' });
        ctx.mod.free(widePtr);
      }
    } else {
      // Temporal types route to their physical kernel token (ADR-010): date32→i32, timestamp→i64.
      const token = DTYPES[col.dtype].wasm;
      callArgsort(wasm, `argsort_${token}`, col.dataPtr, av.ptr, permPtr, len, desc, scratchPtr, needScratch);
    }
  } finally {
    freeAligned(ctx, av);
  }
}

function buildUtf8Rank(rt: DfRuntime, col: Column, len: number): number {
  const { ctx } = rt;
  const strings = col.dict ? decodeDictionary(ctx, col.dict) : [];
  const order = strings.map((_, i) => i).sort((x, y) => {
    const sx = strings[x]!;
    const sy = strings[y]!;
    return sx < sy ? -1 : sx > sy ? 1 : 0;
  });
  const rankOfSlot = new Int32Array(strings.length);
  for (let r = 0; r < order.length; r++) rankOfSlot[order[r]!] = r;

  const rankPtr = ctx.mod.alloc(Math.max(len * 4, 1));
  const idx = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'i32' }) as Int32Array;
  const out = ctx.viewOf({ ptr: rankPtr, length: len, dtype: 'i32' }) as Int32Array;
  for (let i = 0; i < len; i++) {
    const slot = idx[i]!;
    out[i] = slot >= 0 && slot < rankOfSlot.length ? rankOfSlot[slot]! : 0;
  }
  return rankPtr;
}

function widenBoolToI32(rt: DfRuntime, col: Column, len: number): number {
  const { ctx } = rt;
  const outPtr = ctx.mod.alloc(Math.max(len * 4, 1));
  const src = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'bool' }) as Uint8Array;
  const dst = ctx.viewOf({ ptr: outPtr, length: len, dtype: 'i32' }) as Int32Array;
  for (let i = 0; i < len; i++) dst[i] = src[i]!;
  return outPtr;
}
