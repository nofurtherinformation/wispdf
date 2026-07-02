/**
 * Hash join (spec §4; ADR-005). Builds on the right, probes left, via `join_hash_inner`/
 * `join_hash_left` (ABI §9 D; left emits r_idx=-1 → nulls). utf8 keys are dictionary-unified
 * (JS unifyDictionaries) so equal strings across frames share a merged index; bool widens to
 * i32; a null in any key excludes the row. Output = all left columns + right non-key columns
 * (colliding right names suffixed _right).
 */

import { DTYPES } from '../memory/dtype.js';
import { validityBytes, getBit, setBit } from '../memory/bitmap.js';
import { unifyDictionaries } from '../memory/dictionary.js';
import { createColumn, columnToArray, freeColumn, type Cell, type Column, type ColumnInput } from '../memory/column.js';
import { joinHashInner, joinHashLeft, type HashExports, type JoinResult } from '../kernels/hash/index.js';
import type { DfRuntime } from './runtime.js';
import { rawKernel } from './runtime.js';
import { FrameError, unknownColumn, dtypeMismatch } from './errors.js';
import type { GroupBySource, NamedColumn } from './groupby.js';
import type { DataFrame } from './dataframe.js';

export type JoinHow = 'inner' | 'left';

export interface JoinOptions {
  on: string | string[];
  how?: JoinHow;
}

export type JoinSource = GroupBySource;

export function joinFrames(left: JoinSource, right: JoinSource, opts: JoinOptions): DataFrame {
  const how: JoinHow = opts.how ?? 'inner';
  const keys = Array.isArray(opts.on) ? opts.on : [opts.on];
  if (keys.length === 0) throw new FrameError('join requires at least one key in `on`.');

  const rt = left.rt;
  const { ctx } = rt;
  const lLen = left.length;
  const rLen = right.length;

  for (const k of keys) {
    const ld = left.dtypeOf(k);
    if (ld === undefined) throw unknownColumn(k, left.columnNames());
    const rd = right.dtypeOf(k);
    if (rd === undefined) throw unknownColumn(k, right.columnNames());
    if (ld !== rd) throw dtypeMismatch('join', ld, rd, `join key '${k}' must have one dtype.`);
  }

  const prepared = keys.map((k) => prepareKey(rt, left.getColumn(k)!, right.getColumn(k)!, lLen, rLen));

  try {
    const lHash = combinedHash(rt, prepared.map((p) => ({ ptr: p.leftPtr, tok: p.tok })), lLen);
    const rHash = combinedHash(rt, prepared.map((p) => ({ ptr: p.rightPtr, tok: p.tok })), rLen);
    const lvp = combinedValidity(rt, keys.map((k) => left.getColumn(k)!), lLen);
    const rvp = combinedValidity(rt, keys.map((k) => right.getColumn(k)!), rLen);

    let res: JoinResult;
    try {
      const ex = rt.wasm as unknown as HashExports;
      res = how === 'left'
        ? joinHashLeft(ex, lHash, lvp.ptr, lLen, rHash, rvp.ptr, rLen)
        : joinHashInner(ex, lHash, lvp.ptr, lLen, rHash, rvp.ptr, rLen);
    } finally {
      ctx.viewOf.forget({ ptr: lHash, length: lLen, dtype: 'u8' });
      ctx.mod.free(lHash);
      ctx.viewOf.forget({ ptr: rHash, length: rLen, dtype: 'u8' });
      ctx.mod.free(rHash);
      freePtr(rt, lvp.ptr, lvp.owns);
      freePtr(rt, rvp.ptr, rvp.owns);
    }

    return assembleOutput(left, right, keys, res, how);
  } finally {
    for (const p of prepared) p.free();
  }
}

function assembleOutput(
  left: JoinSource,
  right: JoinSource,
  keys: string[],
  res: JoinResult,
  how: JoinHow,
): DataFrame {
  const rt = left.rt;
  const { ctx } = rt;
  const keySet = new Set(keys);
  const leftNames = left.columnNames();
  const rightNames = right.columnNames().filter((n) => !keySet.has(n));
  const leftNameSet = new Set(leftNames);

  const owned: Column[] = [];
  try {
    const out: NamedColumn[] = [];
    for (const name of leftNames) {
      const col = gatherJS(rt, left.getColumn(name)!, res.lIdx);
      owned.push(col);
      out.push({ name, col });
    }
    for (const name of rightNames) {
      const col = gatherJS(rt, right.getColumn(name)!, res.rIdx);
      owned.push(col);
      const outName = leftNameSet.has(name) ? `${name}_right` : name;
      out.push({ name: outName, col });
    }
    const result = left.buildResult(out);
    owned.length = 0;
    return result;
  } finally {
    for (const c of owned) freeColumn(ctx, c);
    void how;
  }
}

function gatherJS(rt: DfRuntime, col: Column, idx: Int32Array): Column {
  const { ctx } = rt;
  const all = columnToArray(ctx, col);
  const out = new Array<Cell>(idx.length);
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k]!;
    out[k] = i < 0 ? null : all[i]!;
  }
  return createColumn(ctx, col.dtype, out as unknown as ColumnInput);
}

interface PreparedKey {
  readonly leftPtr: number;
  readonly rightPtr: number;
  readonly tok: string;
  free(): void;
}

function prepareKey(
  rt: DfRuntime,
  lCol: Column,
  rCol: Column,
  lLen: number,
  rLen: number,
): PreparedKey {
  const { ctx } = rt;
  if (lCol.dtype === 'utf8') {
    const unified = unifyDictionaries(ctx, lCol.dict!, rCol.dict!);
    const leftPtr = remapIndices(rt, lCol, unified.remapA, lLen);
    const rightPtr = remapIndices(rt, rCol, unified.remapB, rLen);
    return {
      leftPtr,
      rightPtr,
      tok: 'i32',
      free() {
        freePtr(rt, leftPtr, true);
        freePtr(rt, rightPtr, true);
      },
    };
  }
  if (lCol.dtype === 'bool') {
    const leftPtr = widenBool(rt, lCol, lLen);
    const rightPtr = widenBool(rt, rCol, rLen);
    return {
      leftPtr,
      rightPtr,
      tok: 'i32',
      free() {
        freePtr(rt, leftPtr, true);
        freePtr(rt, rightPtr, true);
      },
    };
  }
  return { leftPtr: lCol.dataPtr, rightPtr: rCol.dataPtr, tok: DTYPES[lCol.dtype].wasm, free() {} };
}

function remapIndices(rt: DfRuntime, col: Column, remap: Int32Array, len: number): number {
  const { ctx } = rt;
  const outPtr = ctx.mod.alloc(Math.max(len * 4, 1));
  const idx = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'i32' }) as Int32Array;
  const out = ctx.viewOf({ ptr: outPtr, length: len, dtype: 'i32' }) as Int32Array;
  for (let i = 0; i < len; i++) {
    const s = idx[i]!;
    out[i] = s >= 0 && s < remap.length ? remap[s]! : 0;
  }
  return outPtr;
}

function widenBool(rt: DfRuntime, col: Column, len: number): number {
  const { ctx } = rt;
  const outPtr = ctx.mod.alloc(Math.max(len * 4, 1));
  const src = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'bool' }) as Uint8Array;
  const dst = ctx.viewOf({ ptr: outPtr, length: len, dtype: 'i32' }) as Int32Array;
  for (let i = 0; i < len; i++) dst[i] = src[i]!;
  return outPtr;
}

function combinedHash(rt: DfRuntime, inputs: { ptr: number; tok: string }[], len: number): number {
  const { ctx, wasm } = rt;
  const hashPtr = ctx.mod.alloc(Math.max(len * 8, 1));
  for (let k = 0; k < inputs.length; k++) {
    const inp = inputs[k]!;
    if (k === 0) {
      rawKernel(wasm, `hash_${inp.tok}`)(inp.ptr, 0, hashPtr, len);
    } else {
      const tmp = ctx.mod.alloc(Math.max(len * 8, 1));
      try {
        rawKernel(wasm, `hash_${inp.tok}`)(inp.ptr, 0, tmp, len);
        rawKernel(wasm, 'hash_combine')(hashPtr, tmp, len);
      } finally {
        ctx.viewOf.forget({ ptr: tmp, length: len, dtype: 'u8' });
        ctx.mod.free(tmp);
      }
    }
  }
  return hashPtr;
}

function combinedValidity(rt: DfRuntime, cols: Column[], len: number): { ptr: number; owns: boolean } {
  const { ctx } = rt;
  const nullable = cols.filter((c) => c.validityPtr !== 0);
  if (nullable.length === 0) return { ptr: 0, owns: false };
  const vbytes = validityBytes(len);
  const outPtr = ctx.mod.alloc(Math.max(vbytes, 1));
  const dst = ctx.viewOf({ ptr: outPtr, length: Math.max(vbytes, 1), dtype: 'u8' }) as Uint8Array;
  dst.fill(0);
  const srcs = nullable.map((c) => ({
    view: ctx.viewOf({
      ptr: c.validityPtr,
      length: validityBytes(c.validityBitOffset + len),
      dtype: 'u8',
    }) as Uint8Array,
    bitOff: c.validityBitOffset,
  }));
  for (let i = 0; i < len; i++) {
    let ok = true;
    for (const s of srcs) {
      if (!getBit(s.view, s.bitOff + i)) { ok = false; break; }
    }
    if (ok) setBit(dst, i);
  }
  return { ptr: outPtr, owns: true };
}

function freePtr(rt: DfRuntime, ptr: number, owns: boolean): void {
  if (owns && ptr !== 0) {
    rt.ctx.viewOf.forget({ ptr, length: 1, dtype: 'u8' });
    rt.ctx.mod.free(ptr);
  }
}
