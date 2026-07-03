/**
 * JS-side dispatch stubs for the hash kernel family (Phase 2, Agent D).
 *
 * Thin wrappers over the raw WASM exports that implement the two retry
 * protocols defined in wasm-abi.md §9 D (this is Phase 3's contract):
 *
 * 1. **HT-grow protocol** (`group_build`, `join_*`): if the kernel returns
 *    `-1` (hash table full), double `htCap`, re-zero the table, and re-call.
 *
 * 2. **Out-grow protocol** (`join_*`): if the kernel returns `n > outCap`,
 *    the caller should re-allocate `out_l_idx` and `out_r_idx` to `n` and
 *    re-call.  The JS stubs here implement that loop transparently.
 *
 * Both retry paths are tested in `tests/kernels/hash/` (including forced tiny
 * `htCap` / `outCap` variants).
 */

// ---------------------------------------------------------------------------
// WASM export interface (hash + memory allocator)
// ---------------------------------------------------------------------------

/**
 * The subset of WASM exports consumed by the hash family stubs.
 * The actual `WebAssembly.Instance.exports` object is cast to this at
 * load time; all hash_dt, group_build, and join exports live here.
 */
export interface HashExports {
  /** `alloc(size) -> ptr` — 16-byte aligned, 0 on OOM (ABI §3). */
  alloc(size: number): number;
  /** `free(ptr)` — free(0) is a no-op (ABI §3). */
  free(ptr: number): void;
  /** Single linear memory shared by all kernels (ABI §2). */
  readonly memory: WebAssembly.Memory;

  // ── hash_dt ──────────────────────────────────────────────────────────────
  /** `hash_i32(data, vp, out_hash, len) -> ()` */
  hash_i32(data: number, vp: number, outHash: number, len: number): void;
  /** `hash_u32(data, vp, out_hash, len) -> ()` */
  hash_u32(data: number, vp: number, outHash: number, len: number): void;
  /** `hash_f64(data, vp, out_hash, len) -> ()` */
  hash_f64(data: number, vp: number, outHash: number, len: number): void;
  /** `hash_f32(data, vp, out_hash, len) -> ()` */
  hash_f32(data: number, vp: number, outHash: number, len: number): void;
  /** `hash_i64(data, vp, out_hash, len) -> ()` — v2.3 i64 column hash. */
  hash_i64(data: number, vp: number, outHash: number, len: number): void;

  // ── hash_combine ─────────────────────────────────────────────────────────
  /** `hash_combine(acc_hash, add_hash, len) -> ()` — in-place multi-key mix. */
  hash_combine(accHash: number, addHash: number, len: number): void;

  // ── group_build ──────────────────────────────────────────────────────────
  /**
   * `group_build(hash_ptr, len, ht_ptr, ht_cap, out_group_ids) -> i32`
   *
   * Returns `group_count` (≥ 0) or `-1` if the HT is too small.
   */
  group_build(
    hashPtr: number,
    len: number,
    htPtr: number,
    htCap: number,
    outGroupIds: number,
  ): number;

  // ── join_hash_inner / join_hash_left ──────────────────────────────────────
  /**
   * `join_hash_inner / join_hash_left(...) -> i32`
   *
   * Returns total pair count, or `-1` if HT too small.
   */
  join_hash_inner(
    lhPtr: number,
    lVp: number,
    lLen: number,
    rhPtr: number,
    rVp: number,
    rLen: number,
    htPtr: number,
    htCap: number,
    outLIdx: number,
    outRIdx: number,
    outCap: number,
  ): number;
  join_hash_left(
    lhPtr: number,
    lVp: number,
    lLen: number,
    rhPtr: number,
    rVp: number,
    rLen: number,
    htPtr: number,
    htCap: number,
    outLIdx: number,
    outRIdx: number,
    outCap: number,
  ): number;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Next power-of-two ≥ `n`. Returns 1 for n ≤ 1. */
function nextPow2(n: number): number {
  if (n <= 1) return 1;
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/** Minimum initial hash-table capacity (slots). */
const MIN_HT_CAP = 4;

/** Bytes per hash-table slot (16; see hash.rs module docs). */
const SLOT_BYTES = 16;

/**
 * Allocate `bytes` from the WASM heap and zero-initialise.
 * Throws if OOM (`alloc` returned 0).
 *
 * NOTE: alloc may grow `memory` (detaching old TypedArray views). Any
 * view that was built before this call is potentially stale. Always
 * re-read `ex.memory.buffer` after calling this function.
 */
function allocZeroed(ex: HashExports, bytes: number): number {
  if (bytes === 0) {
    // alloc(0) returns a valid non-zero ptr per ABI §3, but no zeroing needed.
    return ex.alloc(0);
  }
  const ptr = ex.alloc(bytes);
  if (ptr === 0) throw new Error(`[hash stubs] OOM: alloc(${bytes}) returned 0`);
  new Uint8Array(ex.memory.buffer, ptr, bytes).fill(0);
  return ptr;
}

// ---------------------------------------------------------------------------
// groupBuild — JS wrapper with HT-grow retry
// ---------------------------------------------------------------------------

/** Result of {@link groupBuild}. */
export interface GroupBuildResult {
  /** Number of distinct groups found. */
  groupCount: number;
  /** `i32[len]` group IDs in first-occurrence order (caller-allocated, caller-frees). */
  outGroupIds: number;
}

/**
 * Assign dense group IDs (first-occurrence order) to a pre-hashed column.
 *
 * Allocates and manages the hash-table internally; retries with doubled
 * capacity on `-1` (HT full). The `outGroupIds` buffer must be caller-
 * allocated (`i32[len]`) and is filled here; it is NOT freed — the caller
 * is responsible for freeing it.
 *
 * @param ex       WASM exports (hash + allocator).
 * @param hashPtr  Pointer to `i64[len]` pre-hashed keys.
 * @param len      Number of rows.
 * @param outGroupIds Pointer to caller-allocated `i32[len]` output buffer.
 * @param htCapHint  Optional initial HT capacity (default: `nextPow2(2*len)`).
 * @returns        `groupCount` — number of distinct groups.
 */
export function groupBuild(
  ex: HashExports,
  hashPtr: number,
  len: number,
  outGroupIds: number,
  htCapHint?: number,
): number {
  let htCap = Math.max(MIN_HT_CAP, htCapHint ?? nextPow2(2 * Math.max(len, 1)));
  let htPtr = 0;
  try {
    htPtr = allocZeroed(ex, htCap * SLOT_BYTES);
    for (;;) {
      const result = ex.group_build(hashPtr, len, htPtr, htCap, outGroupIds);
      if (result !== -1) {
        return result; // success: group_count
      }
      // HT full — double capacity and retry.
      ex.free(htPtr);
      htPtr = 0; // guard against double-free if allocZeroed throws
      htCap *= 2;
      htPtr = allocZeroed(ex, htCap * SLOT_BYTES);
    }
  } finally {
    if (htPtr !== 0) ex.free(htPtr);
  }
}

// ---------------------------------------------------------------------------
// joinHashInner / joinHashLeft — JS wrappers with HT-grow + out-grow retry
// ---------------------------------------------------------------------------

/** Result of {@link joinHashInner} and {@link joinHashLeft}. */
export interface JoinResult {
  /** Number of output pairs. */
  count: number;
  /**
   * `i32[count]` left-row indices.
   * This is a copy of the WASM buffer — safe to hold after further WASM calls.
   */
  lIdx: Int32Array;
  /**
   * `i32[count]` right-row indices.
   * For `joinHashLeft`, unmatched/null-key rows have `rIdx[i] === -1`.
   */
  rIdx: Int32Array;
}

/**
 * Shared implementation for inner and left hash joins.
 *
 * Implements two retry loops:
 * 1. **HT-grow**: if kernel returns `-1`, double `htCap` and re-call.
 * 2. **Out-grow**: if kernel returns `n > outCap`, grow out arrays to `n`
 *    and re-call (the kernel re-runs both build and probe; the HT must be
 *    re-zeroed for the fresh build phase).
 *
 * @param joinFn  Either `ex.join_hash_inner` or `ex.join_hash_left`, bound to `ex`.
 */
function joinImpl(
  ex: HashExports,
  joinFn: (
    lhPtr: number,
    lVp: number,
    lLen: number,
    rhPtr: number,
    rVp: number,
    rLen: number,
    htPtr: number,
    htCap: number,
    outLIdx: number,
    outRIdx: number,
    outCap: number,
  ) => number,
  lhPtr: number,
  lVp: number,
  lLen: number,
  rhPtr: number,
  rVp: number,
  rLen: number,
  htCapHint?: number,
): JoinResult {
  let htCap = Math.max(MIN_HT_CAP, htCapHint ?? nextPow2(2 * Math.max(rLen, 1)));
  // Initial output capacity: upper bound on pairs (every left × every right)
  let outCap = Math.max(1, lLen + rLen);

  let htPtr = 0;
  let outLPtr = 0;
  let outRPtr = 0;

  try {
    htPtr = allocZeroed(ex, htCap * SLOT_BYTES);
    outLPtr = ex.alloc(outCap * 4);
    if (outLPtr === 0) throw new Error('[hash stubs] OOM: out_l_idx alloc failed');
    outRPtr = ex.alloc(outCap * 4);
    if (outRPtr === 0) throw new Error('[hash stubs] OOM: out_r_idx alloc failed');

    for (;;) {
      const n = joinFn(
        lhPtr,
        lVp,
        lLen,
        rhPtr,
        rVp,
        rLen,
        htPtr,
        htCap,
        outLPtr,
        outRPtr,
        outCap,
      );

      if (n === -1) {
        // HT too small — double capacity, re-zero, re-call.
        ex.free(htPtr);
        htPtr = 0;
        htCap *= 2;
        htPtr = allocZeroed(ex, htCap * SLOT_BYTES);
        continue;
      }

      if (n > outCap) {
        // Output too small — grow and re-call (also re-zero HT for fresh build).
        ex.free(outLPtr);
        outLPtr = 0;
        ex.free(outRPtr);
        outRPtr = 0;
        outCap = n;
        outLPtr = ex.alloc(outCap * 4);
        if (outLPtr === 0) throw new Error('[hash stubs] OOM: out_l_idx resize failed');
        outRPtr = ex.alloc(outCap * 4);
        if (outRPtr === 0) throw new Error('[hash stubs] OOM: out_r_idx resize failed');

        // Re-zero the HT: the kernel re-runs build from scratch on re-call.
        ex.free(htPtr);
        htPtr = 0;
        htPtr = allocZeroed(ex, htCap * SLOT_BYTES);
        continue;
      }

      // Success — copy results before freeing the internal WASM buffers.
      const lIdx = new Int32Array(ex.memory.buffer, outLPtr, n).slice();
      const rIdx = new Int32Array(ex.memory.buffer, outRPtr, n).slice();
      return { count: n, lIdx, rIdx };
    }
  } finally {
    // Always release internal HT and output buffers.
    if (htPtr !== 0) ex.free(htPtr);
    if (outLPtr !== 0) ex.free(outLPtr);
    if (outRPtr !== 0) ex.free(outRPtr);
  }
}

/**
 * Inner equi-join (wasm-abi.md §9 D, `join_hash_inner`).
 *
 * Builds on the right side, probes left rows. Null-validity rows (via `l_vp`
 * / `r_vp`) never match. Returns pairs in probe (left-row) order; duplicate
 * right matches are in build (right-row) order.
 */
export function joinHashInner(
  ex: HashExports,
  lhPtr: number,
  lVp: number,
  lLen: number,
  rhPtr: number,
  rVp: number,
  rLen: number,
  htCapHint?: number,
): JoinResult {
  return joinImpl(
    ex,
    (lhP, lV, lL, rhP, rV, rL, htP, htC, olP, orP, oC) =>
      ex.join_hash_inner(lhP, lV, lL, rhP, rV, rL, htP, htC, olP, orP, oC),
    lhPtr,
    lVp,
    lLen,
    rhPtr,
    rVp,
    rLen,
    htCapHint,
  );
}

/**
 * Left outer equi-join (wasm-abi.md §9 D, `join_hash_left`).
 *
 * Same as `joinHashInner` but unmatched left rows AND null-key left rows
 * produce `(l_idx, -1)` pairs. The frame layer maps `-1` right indices to
 * null-gather operations.
 */
export function joinHashLeft(
  ex: HashExports,
  lhPtr: number,
  lVp: number,
  lLen: number,
  rhPtr: number,
  rVp: number,
  rLen: number,
  htCapHint?: number,
): JoinResult {
  return joinImpl(
    ex,
    (lhP, lV, lL, rhP, rV, rL, htP, htC, olP, orP, oC) =>
      ex.join_hash_left(lhP, lV, lL, rhP, rV, rL, htP, htC, olP, orP, oC),
    lhPtr,
    lVp,
    lLen,
    rhPtr,
    rVp,
    rLen,
    htCapHint,
  );
}
