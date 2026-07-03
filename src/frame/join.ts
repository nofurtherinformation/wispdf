/**
 * Hash join (spec §4; ADR-005). Builds on the right, probes left, via `join_hash_inner`/
 * `join_hash_left` (ABI §9 D; left emits r_idx=-1 → nulls). utf8 keys use the
 * unification-free path (ABI §12): each side's dictionary bytes are hashed once via
 * `hash_utf8_dict`, then row hashes are gathered via `gather_i64`; no JS dict unification
 * is performed. bool widens to i32; a null in any key excludes the row. Output = all left
 * columns + right non-key columns (colliding right names suffixed _right). Output utf8 key
 * column reuses the left dictionary (ABI §12).
 */

import { DTYPES } from '../memory/dtype.js';
import { validityBytes, getBit, setBit } from '../memory/bitmap.js';
import { createColumn, columnToArray, freeColumn, type Cell, type Column, type ColumnInput } from '../memory/column.js';
import type { Dictionary } from '../memory/dictionary.js';
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
  const outLen = res.count;
  const keySet = new Set(keys);
  const leftNames = left.columnNames();
  const rightNames = right.columnNames().filter((n) => !keySet.has(n));
  const leftNameSet = new Set(leftNames);

  // Write index arrays to wasm once; reuse for all column gathers.
  // lIdx never contains -1 (every left row appears in both inner and left join).
  // rIdx may contain -1 (unmatched left rows in left join → null right columns).
  const lIdxPtr = idxToWasm(rt, res.lIdx, outLen);
  let rIdxSafePtr: number;
  let rNullMaskBytes: Uint8Array | null = null;
  if (how === 'left') {
    rNullMaskBytes = buildNullMask(res.rIdx, outLen);
    rIdxSafePtr = idxToWasmSafe(rt, res.rIdx, outLen); // -1 → 0
  } else {
    rIdxSafePtr = idxToWasm(rt, res.rIdx, outLen);
  }

  const owned: Column[] = [];
  try {
    const out: NamedColumn[] = [];
    for (const name of leftNames) {
      const col = gatherWasm(rt, left.getColumn(name)!, lIdxPtr, outLen, null);
      owned.push(col);
      out.push({ name, col });
    }
    for (const name of rightNames) {
      const col = gatherWasm(rt, right.getColumn(name)!, rIdxSafePtr, outLen, rNullMaskBytes);
      owned.push(col);
      const outName = leftNameSet.has(name) ? `${name}_right` : name;
      out.push({ name: outName, col });
    }
    const result = left.buildResult(out);
    owned.length = 0;
    return result;
  } finally {
    for (const c of owned) freeColumn(ctx, c);
    ctx.mod.free(lIdxPtr);
    ctx.mod.free(rIdxSafePtr);
    void how;
  }
}

// ---------------------------------------------------------------------------
// Wasm-based gather helpers (fast path for assembleOutput)
// ---------------------------------------------------------------------------

/** Write a JS Int32Array to wasm. Caller frees the returned pointer. */
function idxToWasm(rt: DfRuntime, idx: Int32Array, len: number): number {
  const ptr = rt.ctx.mod.alloc(Math.max(len * 4, 1));
  if (len > 0) new Int32Array(rt.wasm.memory.buffer, ptr, len).set(idx);
  return ptr;
}

/** Like idxToWasm but replaces -1 with 0 (safe sentinel for wasm gather kernels). */
function idxToWasmSafe(rt: DfRuntime, idx: Int32Array, len: number): number {
  const ptr = rt.ctx.mod.alloc(Math.max(len * 4, 1));
  if (len > 0) {
    const view = new Int32Array(rt.wasm.memory.buffer, ptr, len);
    for (let k = 0; k < len; k++) view[k] = idx[k]! < 0 ? 0 : idx[k]!;
  }
  return ptr;
}

/**
 * Build an Arrow-LSB null-mask byte array (JS-side) from rIdx.
 * Bit k = 1 when rIdx[k] ≥ 0 (valid), 0 when rIdx[k] = -1 (null).
 * Returns null if there are no -1 values (all rows matched).
 */
function buildNullMask(rIdx: Int32Array, outLen: number): Uint8Array | null {
  let hasNeg = false;
  for (let k = 0; k < outLen; k++) {
    if (rIdx[k]! < 0) { hasNeg = true; break; }
  }
  if (!hasNeg) return null;
  const vb = validityBytes(outLen);
  const mask = new Uint8Array(vb).fill(0xff);
  for (let k = 0; k < outLen; k++) {
    if (rIdx[k]! < 0) { const bi = k >> 3; mask[bi] = mask[bi]! & ~(1 << (k & 7)); }
  }
  if (outLen & 7) { mask[vb - 1] = mask[vb - 1]! & ((1 << (outLen & 7)) - 1); }
  return mask;
}

/** Create an owned root Column from raw wasm pointers (no data copy). */
function makeRootCol(
  dtype: Column['dtype'],
  length: number,
  dataPtr: number,
  validityPtr: number,
  dict: Dictionary | null,
): Column {
  return { dtype, length, dataPtr, validityPtr, validityBitOffset: 0, dict, owned: true };
}

/**
 * Byte-level dictionary clone: copies offsets + bytes buffers in wasm without
 * decoding any strings. Both allocs happen before any TypedArray views are taken
 * so memory growth cannot produce stale views.
 */
function cloneDictBytes(rt: DfRuntime, dict: Dictionary): Dictionary {
  const { count, offsetsPtr, bytesPtr, bytesLen } = dict;
  const offsSize = (count + 1) * 4;
  const newOffsetsPtr = rt.ctx.mod.alloc(offsSize);
  const newBytesPtr   = rt.ctx.mod.alloc(Math.max(bytesLen, 1));
  const buf = rt.wasm.memory.buffer; // post-growth buffer after both allocs
  new Uint8Array(buf, newOffsetsPtr, offsSize).set(new Uint8Array(buf, offsetsPtr, offsSize));
  if (bytesLen > 0) {
    new Uint8Array(buf, newBytesPtr, bytesLen).set(new Uint8Array(buf, bytesPtr, bytesLen));
  }
  return { count, offsetsPtr: newOffsetsPtr, bytesPtr: newBytesPtr, bytesLen };
}

/**
 * Build the output validity bitmap for a gathered column.
 *
 * Output row k is valid iff:
 *   (a) nullMask is null OR bit k of nullMask is 1 (not a forced-null from -1 rIdx), AND
 *   (b) source validity at col.validityPtr[idx[k]] is 1 (or col has no nulls).
 *
 * Returns 0 (all-valid sentinel) when no nulls need encoding.
 * Uses the `gather_validity` wasm kernel to propagate source nulls efficiently.
 */
function buildGatherValidity(
  rt: DfRuntime,
  col: Column,
  idxPtr: number,
  outLen: number,
  nullMask: Uint8Array | null,
): number {
  const srcHasNulls = col.validityPtr !== 0;
  const hasForcedNulls = nullMask !== null;
  if (!srcHasNulls && !hasForcedNulls) return 0;

  const vb = validityBytes(outLen);

  if (!hasForcedNulls) {
    // Only source nulls: delegate entirely to wasm.
    const vp = rt.ctx.mod.alloc(Math.max(vb, 1));
    rawKernel(rt.wasm, 'gather_validity')(col.validityPtr, idxPtr, outLen, vp);
    return vp;
  }

  // Has forced nulls: start from nullMask, AND in source validity if needed.
  const vp = rt.ctx.mod.alloc(Math.max(vb, 1));
  new Uint8Array(rt.wasm.memory.buffer, vp, vb).set(nullMask!);

  if (srcHasNulls) {
    const tmp = rt.ctx.mod.alloc(Math.max(vb, 1));
    try {
      rawKernel(rt.wasm, 'gather_validity')(col.validityPtr, idxPtr, outLen, tmp);
      // Re-read after tmp alloc (may have grown memory).
      const dst = new Uint8Array(rt.wasm.memory.buffer, vp,  vb);
      const src = new Uint8Array(rt.wasm.memory.buffer, tmp, vb);
      for (let b = 0; b < vb; b++) dst[b] = dst[b]! & src[b]!;
    } finally {
      rt.ctx.mod.free(tmp);
    }
  }
  return vp;
}

// Actual gather kernel token: bool is stored as u8, so gather_u8 not gather_bool.
// ponytail: only exception; all other wasm tokens map directly to DTYPES[d].wasm.
const GATHER_TOKEN: Partial<Record<string, string>> = { bool: 'u8' };

/**
 * Fast wasm-based column gather for join output.
 *
 * Numeric/bool/i64/date32/timestamp: wasm `gather_*` kernel; no JS boxing.
 * utf8: wasm `gather_i32` for dict indices + byte-level dict clone; no string decode.
 *
 * @param idxPtr  Wasm ptr to i32[outLen] source indices (no -1; caller sanitises).
 * @param nullMask  Arrow-LSB forced-null bitmap for left-join right-side; null = no forced nulls.
 */
function gatherWasm(rt: DfRuntime, col: Column, idxPtr: number, outLen: number, nullMask: Uint8Array | null): Column {
  const { ctx } = rt;

  if (col.dtype === 'utf8') {
    const outIdxPtr = ctx.mod.alloc(Math.max(outLen * 4, 1));
    if (outLen > 0) rawKernel(rt.wasm, 'gather_i32')(col.dataPtr, idxPtr, outLen, outIdxPtr);
    const dict = cloneDictBytes(rt, col.dict!);
    return makeRootCol('utf8', outLen, outIdxPtr, buildGatherValidity(rt, col, idxPtr, outLen, nullMask), dict);
  }

  const info = DTYPES[col.dtype];
  const tok  = GATHER_TOKEN[info.wasm] ?? info.wasm;
  const outDataPtr = ctx.mod.alloc(Math.max(outLen * info.size, 1));
  if (outLen > 0) rawKernel(rt.wasm, `gather_${tok}`)(col.dataPtr, idxPtr, outLen, outDataPtr);
  return makeRootCol(col.dtype, outLen, outDataPtr, buildGatherValidity(rt, col, idxPtr, outLen, nullMask), null);
}

// Keep gatherJS as a fallback for any edge case or for testing.
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
void gatherJS; // suppress unused warning — kept as reference/fallback

/**
 * Sentinel token for pre-computed i64[len] row hashes (ABI §12 utf8 path).
 * Used in `combinedHash` to distinguish pre-hashed buffers from raw column data.
 */
const TOK_I64_DIRECT = 'i64_direct' as const;

interface PreparedKey {
  readonly leftPtr: number;
  readonly rightPtr: number;
  /** Dtype token for `hash_${tok}`, or `TOK_I64_DIRECT` for pre-hashed i64[len]. */
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
  if (lCol.dtype === 'utf8') {
    // Unification-free utf8 path (ABI §12):
    //   hash_utf8_dict(dict) → i64[dict_count] per side (raw UTF-8 byte hashing)
    //   gather_i64(dictHashes, indices, len) → i64[len] per side (expand to row level)
    // Equal strings produce equal hashes regardless of their slot position in
    // each side's dictionary, so no cross-dictionary remapping is needed.
    const leftPtr = hashUtf8RowHashes(rt, lCol, lLen);
    const rightPtr = hashUtf8RowHashes(rt, rCol, rLen);
    return {
      leftPtr,
      rightPtr,
      tok: TOK_I64_DIRECT,
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

/**
 * Hash the UTF-8 bytes of a utf8 column's dictionary once, then gather per-row
 * hashes via `gather_i64`.  Returns a caller-owned pointer to `i64[len]`.
 *
 * Steps (ABI §12):
 *   1. Alloc `i64[dictCount]`; call `hash_utf8_dict` to hash raw bytes per slot.
 *   2. Alloc `i64[len]`; call `gather_i64(dictHashes, col.dataPtr, len)` to expand.
 *   3. Free the temporary dict-hash buffer.
 *   4. Return the row-hash buffer (caller frees via `prepareKey.free()`).
 *
 * Null rows have their validity handled by the join kernel's `l_vp`/`r_vp` bitmaps;
 * their gathered hash value is immaterial.
 */
function hashUtf8RowHashes(rt: DfRuntime, col: Column, len: number): number {
  const { ctx, wasm } = rt;
  const dict = col.dict!;
  const dictCount = dict.count;

  // Allocate i64[dictCount] for per-slot dict hashes.
  const dictHashPtr = ctx.mod.alloc(Math.max(dictCount * 8, 1));
  try {
    // hash_utf8_dict(offsets_ptr, bytes_ptr, dict_count, out_hash_ptr)
    rawKernel(wasm, 'hash_utf8_dict')(dict.offsetsPtr, dict.bytesPtr, dictCount, dictHashPtr);

    // Allocate i64[len] for per-row hashes.
    const rowHashPtr = ctx.mod.alloc(Math.max(len * 8, 1));
    if (dictCount === 0 || len === 0) {
      // Nothing to gather; zero-init is fine (null rows excluded by validity anyway).
      return rowHashPtr;
    }
    // gather_i64(data: i64[], idx: i32[], idx_len, out: i64[])
    // data = dict hashes, idx = column indices (i32[len]), out = row hashes
    rawKernel(wasm, 'gather_i64')(dictHashPtr, col.dataPtr, len, rowHashPtr);
    return rowHashPtr;
  } finally {
    // Free the temporary dict-hash scratch; no viewOf was taken on it.
    ctx.mod.free(dictHashPtr);
  }
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
  // Allocate the accumulator hash buffer i64[len].  This alloc may grow WASM memory;
  // all TypedArray views taken below use memory.buffer AFTER this call.
  const hashPtr = ctx.mod.alloc(Math.max(len * 8, 1));
  const byteLen = len * 8;

  for (let k = 0; k < inputs.length; k++) {
    const inp = inputs[k]!;
    if (inp.tok === TOK_I64_DIRECT) {
      // Pre-computed i64[len] row hashes (utf8 unification-free path, ABI §12).
      if (k === 0) {
        // Copy pre-computed hashes into the accumulator.  Both pointers are valid in
        // the current memory.buffer (all allocs are done; kernels never grow memory).
        if (byteLen > 0) {
          const buf = wasm.memory.buffer;
          new Uint8Array(buf, hashPtr, byteLen).set(new Uint8Array(buf, inp.ptr, byteLen));
        }
      } else {
        // Multi-key: combine pre-computed hashes into the accumulator in-place.
        rawKernel(wasm, 'hash_combine')(hashPtr, inp.ptr, len);
      }
    } else if (k === 0) {
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
