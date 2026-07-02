/** Column gather (take-by-index) via `gather_dt` + `gather_validity` (ABI §9 C): builds a
 * fresh owned column whose row k is source row idx[k]. Used by sortValues and filterFn. */

import { validityBytes } from '../memory/bitmap.js';
import type { Column } from '../memory/column.js';
import type { DfRuntime } from './runtime.js';
import { rawKernel } from './runtime.js';
import {
  storageToken,
  alignValidity,
  freeAligned,
  copyDictionary,
  dataBytes,
} from './util.js';

export function gatherColumn(
  rt: DfRuntime,
  col: Column,
  idxPtr: number,
  idxLen: number,
): Column {
  const { ctx, wasm } = rt;
  const tok = storageToken(col.dtype);
  const outData = ctx.mod.alloc(dataBytes(col.dtype, idxLen));
  rawKernel(wasm, `gather_${tok}`)(col.dataPtr, idxPtr, idxLen, outData);

  let validityPtr = 0;
  if (col.validityPtr !== 0) {
    const src = alignValidity(ctx, col);
    const vbytes = validityBytes(idxLen);
    const outVp = ctx.mod.alloc(Math.max(vbytes, 1));
    (ctx.viewOf({ ptr: outVp, length: Math.max(vbytes, 1), dtype: 'u8' }) as Uint8Array).fill(0);
    rawKernel(wasm, 'gather_validity')(src.ptr, idxPtr, idxLen, outVp);
    freeAligned(ctx, src);
    validityPtr = outVp;
  }

  const dict = col.dict ? copyDictionary(ctx, col.dict) : null;
  return {
    dtype: col.dtype,
    length: idxLen,
    dataPtr: outData,
    validityPtr,
    validityBitOffset: 0,
    dict,
    owned: true,
  };
}
