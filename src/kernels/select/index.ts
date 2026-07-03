/**
 * JS-side dispatch for the select kernel family (Phase 2 / ABI v1.2).
 *
 * ## argsort (ABI v1.2)
 * Wasm export signature: argsort_dt(data, vp, inout_perm, len, desc, scratch_ptr).
 * The `scratch_ptr` is a caller-allocated `i32[len]` merge scratch. This layer
 * allocates and frees it around every wasm call so neither the kernel nor Phase 3
 * callers need to manage it.
 *
 * TS-facing API: argsort(ctx, col, opts?) → Int32Array. This signature is stable
 * for Phase 3 consumers.
 *
 * ## filter_indices (ABI v1.2 dispatch note)
 * The wasm `filter_indices` export is kept compiled (for completeness and any
 * direct-wasm callers), but this dispatch layer uses a pure-JS implementation.
 * Rationale per ABI v1.2 §9: V8's JIT beats wasm on this bitmap→index scatter
 * loop (measured 0.59× wasm/JS); Math.clz32(m & -m) ^ 31 lowers to native ctz
 * and JS avoids the wasm↔JS boundary overhead on every bit-extraction step.
 */

import type { MemoryContext } from '../../memory/context.js';
import type { Column } from '../../memory/column.js';

// ── Wasm module interface (v1.2 signatures) ───────────────────────────────────

/** Minimum wasm exports required by this dispatch layer. */
export interface SelectWasm {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: (...args: any[]) => any;
  readonly memory: WebAssembly.Memory;
  alloc(size: number): number;
  free(ptr: number): void;
  /** v1.2: data, vp, inout_perm, len, desc, scratch_ptr */
  argsort_f64(data: number, vp: number, perm: number, len: number, desc: number, scratch: number): void;
  argsort_f32(data: number, vp: number, perm: number, len: number, desc: number, scratch: number): void;
  argsort_i32(data: number, vp: number, perm: number, len: number, desc: number, scratch: number): void;
  argsort_u32(data: number, vp: number, perm: number, len: number, desc: number, scratch: number): void;
  argsort_i64(data: number, vp: number, perm: number, len: number, desc: number, scratch: number): void;
  /** Kept compiled; not called by this dispatch layer (ABI v1.2). */
  filter_indices(mask: number, out_idx: number, len: number): number;
}

// ── argsort ───────────────────────────────────────────────────────────────────

export interface ArgsortOpts {
  /** Sort descending. Default false (ascending). */
  descending?: boolean;
}

/**
 * Stably argsort a numeric column, returning the permutation as an `Int32Array`.
 *
 * Total order per dtypes.md §4.6: valid values first (in numeric / IEEE order),
 * NaN after +inf (ascending) / first (descending), nulls always last.
 *
 * Allocates and frees both the output permutation (wasm) and the merge scratch
 * (ABI v1.2) around the wasm call. Returns a copied JS-owned `Int32Array` — the
 * caller does not need to manage any wasm memory.
 *
 * TS-facing API for Phase 3: `argsort(ctx, col, opts?) → Int32Array`.
 */
export function argsort(
  ctx: MemoryContext,
  col: Column,
  opts?: ArgsortOpts,
): Int32Array {
  const { mod } = ctx;
  const n = col.length;
  const desc = opts?.descending ? 1 : 0;
  const dtype = col.dtype as 'f64' | 'f32' | 'i32' | 'u32';

  if (n === 0) {
    return new Int32Array(0);
  }

  // Allocate output permutation (i32[n]) and scratch (i32[n])
  const permPtr = mod.alloc(n * 4);
  if (permPtr === 0) throw new Error('argsort: OOM allocating perm');

  const scratchPtr = mod.alloc(n * 4);
  if (scratchPtr === 0) {
    mod.free(permPtr);
    throw new Error('argsort: OOM allocating scratch');
  }

  // Initialize permutation to identity [0, 1, 2, ..., n-1]
  const permView = new Int32Array(mod.memory.buffer, permPtr, n);
  for (let i = 0; i < n; i++) permView[i] = i;

  // Call the dtype-specific wasm argsort (ABI v1.2 signature)
  if (n > 1) {
    const wasmFn = (mod as unknown as SelectWasm)[`argsort_${dtype}`] as
      (data: number, vp: number, perm: number, len: number, desc: number, scratch: number) => void;
    wasmFn(col.dataPtr, col.validityPtr, permPtr, n, desc, scratchPtr);
  }

  // Copy result to a JS-owned buffer before freeing wasm memory
  // Re-read the view in case memory grew during the call (generation counter)
  const result = new Int32Array(new Int32Array(mod.memory.buffer, permPtr, n));

  mod.free(scratchPtr);
  mod.free(permPtr);

  return result;
}

// ── filter_indices (JS implementation) ───────────────────────────────────────

/**
 * Convert an Arrow-LSB bitmask to an `Int32Array` of set row indices.
 *
 * Pure-JS implementation per ABI v1.2 §9 dispatch note: V8's JIT outperforms
 * the wasm export for this bitmap→index scatter loop. `Math.clz32(m & -m) ^ 31`
 * computes the lowest-set-bit position (equivalent to ctz) and lowers to a
 * native BSF/TZCNT in V8. The JS engine also eliminates the wasm↔JS boundary
 * cost paid on every bit-group iteration.
 *
 * The wasm `filter_indices` export remains compiled but is NOT called here.
 *
 * @param mask - Arrow-LSB bitmask (1 bit per element, LSB-first per byte)
 * @param len  - element count (≤ mask.length * 8)
 * @returns `Int32Array` of indices where the mask bit is `1`
 */
export function filter_indices(mask: Uint8Array, len: number): Int32Array {
  const out = new Int32Array(len); // worst-case: all selected
  let count = 0;
  const full = len >> 3;   // complete mask bytes
  const tail = len & 7;    // remaining elements in last partial byte

  for (let b = 0; b < full; b++) {
    let m = mask[b]!;
    const base = b << 3;
    while (m !== 0) {
      // Lowest-set-bit index = ctz(m) = clz(m & -m) ^ 31
      out[count++] = base + (Math.clz32(m & -m) ^ 31);
      m &= m - 1; // clear lowest set bit
    }
  }

  if (tail > 0) {
    // Mask off padding bits in the final partial byte
    let m = mask[full]! & ((1 << tail) - 1);
    const base = full << 3;
    while (m !== 0) {
      out[count++] = base + (Math.clz32(m & -m) ^ 31);
      m &= m - 1;
    }
  }

  return out.subarray(0, count);
}

// ── Low-level pointer-based dispatch stubs ────────────────────────────────────
// These mirror the elementwise pattern for callers that manage their own buffers.

/** filter_dt: compact data where mask bit = 1; returns output element count. */
export const filter_f64 = (w: SelectWasm, data: number, mask: number, out: number, len: number): number =>
  w.filter_f64(data, mask, out, len);
export const filter_f32 = (w: SelectWasm, data: number, mask: number, out: number, len: number): number =>
  w.filter_f32(data, mask, out, len);
export const filter_i32 = (w: SelectWasm, data: number, mask: number, out: number, len: number): number =>
  w.filter_i32(data, mask, out, len);
export const filter_u32 = (w: SelectWasm, data: number, mask: number, out: number, len: number): number =>
  w.filter_u32(data, mask, out, len);
export const filter_u8  = (w: SelectWasm, data: number, mask: number, out: number, len: number): number =>
  w.filter_u8(data, mask, out, len);

/** gather_dt: out[k] = data[idx[k]] for k in 0..idx_len. */
export const gather_f64 = (w: SelectWasm, data: number, idx: number, idxLen: number, out: number): void =>
  w.gather_f64(data, idx, idxLen, out);
export const gather_f32 = (w: SelectWasm, data: number, idx: number, idxLen: number, out: number): void =>
  w.gather_f32(data, idx, idxLen, out);
export const gather_i32 = (w: SelectWasm, data: number, idx: number, idxLen: number, out: number): void =>
  w.gather_i32(data, idx, idxLen, out);
export const gather_u32 = (w: SelectWasm, data: number, idx: number, idxLen: number, out: number): void =>
  w.gather_u32(data, idx, idxLen, out);
export const gather_u8  = (w: SelectWasm, data: number, idx: number, idxLen: number, out: number): void =>
  w.gather_u8(data, idx, idxLen, out);
export const gather_validity = (w: SelectWasm, vp: number, idx: number, idxLen: number, outVp: number): void =>
  w.gather_validity(vp, idx, idxLen, outVp);

/** topk_dt: k extreme valid indices; returns count written. */
export const topk_f64 = (w: SelectWasm, data: number, vp: number, k: number, outIdx: number, len: number, largest: number): number =>
  w.topk_f64(data, vp, k, outIdx, len, largest);
export const topk_f32 = (w: SelectWasm, data: number, vp: number, k: number, outIdx: number, len: number, largest: number): number =>
  w.topk_f32(data, vp, k, outIdx, len, largest);
export const topk_i32 = (w: SelectWasm, data: number, vp: number, k: number, outIdx: number, len: number, largest: number): number =>
  w.topk_i32(data, vp, k, outIdx, len, largest);
export const topk_u32 = (w: SelectWasm, data: number, vp: number, k: number, outIdx: number, len: number, largest: number): number =>
  w.topk_u32(data, vp, k, outIdx, len, largest);

// ── i64 select stubs (v2.3) ──────────────────────────────────────────────────

export const filter_i64 = (w: SelectWasm, data: number, mask: number, out: number, len: number): number =>
  w.filter_i64(data, mask, out, len);
export const gather_i64 = (w: SelectWasm, data: number, idx: number, idxLen: number, out: number): void =>
  w.gather_i64(data, idx, idxLen, out);
export const topk_i64   = (w: SelectWasm, data: number, vp: number, k: number, outIdx: number, len: number, largest: number): number =>
  w.topk_i64(data, vp, k, outIdx, len, largest);
