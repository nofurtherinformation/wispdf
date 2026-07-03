# WASM ABI — v1 (orchestrator-owned contract)

**Status:** v1, authoritative. This document is the contract between the JS API/memory
layer and the Rust kernel layer (ADR-007 = Rust, ADR-008 = stable flat ABI). Kernel
subagents (Phase 2) and the memory agent (Phase 1) code **against this file**. It is
read-only to subagents; changes require an orchestrator edit + ADR if a locked decision
is affected.

Cross-references: ADR-001 (memory ownership), ADR-002 (columnar layout), ADR-004 (dual
SIMD/scalar builds), ADR-005 (hash relational ops), ADR-007 (Rust), ADR-008 (flat ABI).
Type/cast/null rules live in `contracts/dtypes.md`.

---

## 1. Execution & language model

- One Rust crate per `wasm/` (ADR-007), `crate-type = ["cdylib"]`,
  `target = wasm32-unknown-unknown`, `#![no_std]`, `panic = "abort"`.
- Two binaries per ADR-004, **same source**: `scalar.wasm` (default) and `simd.wasm`
  (built with `RUSTFLAGS="-C target-feature=+simd128"`). SIMD code paths are gated by
  `#[cfg(target_feature = "simd128")]` in one source file per kernel family. Both
  binaries are post-processed with `wasm-opt -O3` (`--enable-simd` on the SIMD build).
- Every exported symbol below exists in **both** binaries with an identical signature
  and identical observable behavior. JS feature-detects and loads one binary; it must
  never depend on which one is loaded except for speed.
- **Size discipline (hard requirement, ADR-007):** stay `no_std` + `panic="abort"`; do
  not pull in `std`, formatting, or unwinding. CI size gate: each `*.wasm` ≤ 75 KB
  gzipped.

## 2. Memory ownership (ADR-001)

- The module exports a single `WebAssembly.Memory` as `memory`. **All** column data,
  validity bitmaps, dictionaries, index buffers, and kernel scratch live inside this
  linear memory. JS never owns column bytes — it holds `TypedArray` views over
  `memory.buffer` (zero-copy in both directions).
- `memory.grow` **detaches every JS view**. Only the allocator (§3) may grow memory,
  and only between kernel calls (see §5). Kernels never grow memory.
- Pointers are **`i32` byte offsets** into linear memory (wasm32). `0` is the null
  pointer and is never a valid data address (the allocator never returns `0` on
  success).

### Generation-counter protocol (invalidation)

- The module maintains a monotonically increasing **generation counter**, exported as:
  - `mem_generation() -> i32` — returns the current generation. Its value **changes**
    (increments) on every successful `memory.grow`.
  - (Phase 1 may additionally export it as a mutable global for cheaper reads; the
    function is the required contract.)
- JS side: a single `viewOf(column)` accessor caches `(generation, view)`. Before use,
  it compares `mem_generation()` to the cached generation; on mismatch it rebuilds all
  views over the **current** `memory.buffer` and updates the cache. **No raw
  `TypedArray` is cached anywhere else** (ADR-001). This is the only sanctioned way to
  hold a view across a call that might have grown memory.

## 3. Allocator exports (Phase 1 owns implementation; ABI fixed here)

The Phase-0 spike shipped a static bump allocator (`alloc` only). Phase 1 replaces it
with a real arena (bump + freelist) exporting:

| Export | Signature (wasm) | Semantics |
|---|---|---|
| `alloc` | `(i32 size) -> i32` | Allocate `size` bytes. Returns a **16-byte-aligned** pointer, or `0` on OOM (a `memory.grow` that failed). `alloc(0)` returns a valid aligned pointer that must not be dereferenced. |
| `free` | `(i32 ptr) -> ()` | Free a block previously returned by `alloc`/`realloc`. `free(0)` is a no-op. Double-free is undefined (Phase-1 tests guard it). |
| `realloc` | `(i32 ptr, i32 new_size) -> i32` | Resize. Returns a 16-byte-aligned pointer (may differ from `ptr`; old contents preserved up to `min(old,new)`), or `0` on OOM (original block stays valid on failure). `realloc(0, n)` ≡ `alloc(n)`. |

**Alignment guarantee: 16 bytes** for every non-null allocation. This makes aligned
`v128` loads legal and gives natural alignment to every v1 dtype (f64=8, f32/i32/u32=4,
u8/bool=1). Kernels may assume 16-byte-aligned base pointers for column buffers.

`alloc`/`realloc` are the **only** functions that call `memory.grow`, and they bump
the generation counter (§2) when they do.

## 4. Buffer conventions

### 4.1 Validity bitmap (Arrow LSB, 1 = valid)

- 1 bit per element, **LSB-first** within each byte. Element `i` is **valid** iff
  `bitmap[i >> 3] & (1 << (i & 7))` is nonzero. `1 = valid`, `0 = null`.
- Length in bytes = `ceil(len / 8)`. Bits beyond `len` in the final byte are **padding**;
  kernels must not depend on them and should write them as `0` when producing a bitmap.
- **All-valid shortcut:** a null-aware kernel receiving `validity_ptr == 0` treats the
  column as **all valid** and may take its fastest path. Callers pass `0` for columns
  with no nulls (the common case). A non-zero `validity_ptr` always points to a real
  `ceil(len/8)`-byte bitmap.

### 4.2 Internal comparison mask

Comparison kernels emit a **mask** in the exact same bit layout as a validity bitmap
(1 bit/element, Arrow LSB, `1 = predicate true`). `filter` consumes a mask directly.
This mask is distinct from a materialized **boolean column** (see §4.4).

### 4.3 Column data buffers

- Numeric/bool data is a contiguous array of the dtype's storage unit, at a
  16-byte-aligned base pointer. `len` in every signature is the **element count**, not a
  byte count; kernels compute byte offsets internally (`i * sizeof(dtype)`).
- `bool` column data is `u8`, one byte per value (`0`/`1`) — see `dtypes.md`. (This is
  the value storage; validity is still a separate bitmap. Internal 1-bit masks are §4.2,
  not bool columns.)

### 4.4 Dictionary-encoded strings (`utf8`, ADR-002)

A `utf8` column is **three** buffers in linear memory:

1. **indices** — `i32[len]`, one dictionary index per row (this is the column's data
   buffer). Indices are `≥ 0`. Null rows are marked in the column's **validity bitmap**,
   not by a sentinel index.
2. **offsets** — `i32[dict_count + 1]`, Arrow-style monotonic non-decreasing byte
   offsets into `bytes`; `offsets[0] == 0`; string `k` occupies `bytes[offsets[k] ..
   offsets[k+1])`.
3. **bytes** — `u8[offsets[dict_count]]`, UTF-8 encoded concatenation of the
   `dict_count` unique strings.

Kernels operate on the **`i32` indices** for compare/hash/group/sort; the dictionary
itself is touched only by dictionary build/unification (Phase 2 Agent D) and by JS
decode (memoized per slot, ADR-002). Dictionary **unification** remaps one column's
indices into a merged dictionary before cross-column relational ops.

## 5. Calling convention (ADR-008)

1. **Flat C exports.** Every kernel is `#[no_mangle] pub unsafe extern "C" fn ...`.
   Parameters and returns are wasm value types only: `i32` (pointers, element counts,
   dictionary indices, bool-as-`i32`), `i64` (64-bit hashes), `f32`/`f64` (scalar
   operands/reduction results). No structs by value across the boundary.
2. **Arrays as (ptr, len).** `ptr` is a byte offset; `len` is element count. Multi-buffer
   inputs pass one `ptr` each.
3. **Outputs via caller-allocated out-params** (`out_ptr`, `out_mask_ptr`,
   `out_validity_ptr`), **except** scalar reductions which return the scalar directly
   (e.g. `sum_f64_null -> f64`). A reduction producing several scalars writes them to a
   caller-provided out array.
4. **Caller owns all buffers.** The JS layer allocates every input, output, and any
   scratch the signature names, *before* the call. **Kernels never `alloc`, never
   `free`, never grow memory, and hold no mutable global state.** Consequence: every
   `TypedArray` view passed into a kernel stays valid for that call (memory cannot grow
   mid-kernel). Any bounded scratch a kernel needs is stack-local.
5. **Null-aware kernels** take a `validity_ptr` (`0` = all valid, §4.1). Kernels that
   produce nullable output also take an `out_validity_ptr` and write it. Elementwise
   arithmetic is **data-only** (see §6) — validity is combined by a separate bitmap
   kernel so the arithmetic path stays branchless and SIMD-friendly. (This refines the
   illustrative signature in ADR-008; ADR-008 delegates the exact parameter lists to
   this document.)
6. **Determinism.** Given identical inputs, both builds produce byte-identical outputs
   (modulo IEEE-754 float reduction order, which must be fixed per kernel and documented
   — reductions specify their accumulation strategy so scalar and SIMD agree within
   spec). No RNG, no time, no I/O.

## 6. Kernel naming scheme

Grammar: **`[family_]op_dtype[_variant]`**, all lowercase, `snake_case`.

- **`op`** — the operation: `add sub mul div mod neg`, `gt ge lt le eq ne`,
  `and or not`, `sum mean min max count nunique std var first last`, `cast fill_null
  is_null`, `gather filter argsort topk`, `hash group join`, `validity_and validity_or`,
  `unify` (dictionaries), `expand_mask` (mask→bool column), etc.
- **`dtype`** — operand storage dtype: `f64 f32 i32 u32 bool u8 utf8`. Casts encode both:
  `cast_<from>_<to>` (e.g. `cast_f64_i32`).
- **`variant`** (optional) — disambiguating suffix: `_null` (null-aware), `_mask`
  (bit-mask output), `_scalar` (scalar right-hand operand), `_s`/`_u` (signed/unsigned
  where the dtype alone is ambiguous), `_stable` (stable sort), etc.
- **`family`** prefix — the elementwise arithmetic/comparison/boolean ops omit it
  (their op names are globally unique: `add_f64`, `gt_f64_mask`). Families whose op
  names would collide carry it (`hash_i32`, `group_*`, `join_*`). The family also
  determines the source path (`wasm/**/elementwise*`, `reduce*`, `select*`, `hash*`).

Examples: `add_f64`, `sub_i32`, `gt_f64_mask`, `sum_f64_null`, `mean_f32_null`,
`cast_f64_i32`, `gather_f64`, `argsort_i32_stable`, `hash_i32`, `validity_and`.

## 7. Trap / error policy

- **Kernels do not validate arguments.** No bounds checks on `ptr`/`len`/alignment; the
  JS API layer guarantees valid, in-bounds, correctly-aligned pointers by construction.
  This ABI is an internal boundary, not the public API. An out-of-bounds access **traps**
  (surfaces to JS as `WebAssembly.RuntimeError`); such a trap is a library bug in the
  caller, not an expected error path.
- **`panic = "abort"`**: any Rust panic becomes an `unreachable` trap. Kernels must be
  written so they cannot panic on valid inputs (use unchecked indexing on pointers, keep
  invariants). Never rely on panic for control flow.
- **Integer divide/mod by zero:** `div_i32/div_u32/mod_i32/mod_u32` must **not trap** on
  a zero divisor. Per `dtypes.md`, a zero divisor yields a **null** result (clear the
  output validity bit; output data value is unspecified but must be written, typically
  `0`). Kernels guard the divisor rather than executing `i32.div_s` on `0`.
- **Float divide by zero / overflow:** follow IEEE-754 (`±inf`, `NaN`); never trap.
- **NaN vs null:** null is tracked **only** by the validity bitmap. A genuine `NaN`
  (validity bit = 1, data = `NaN`) is a *valid* value; comparisons against it follow
  IEEE (all false except `ne`), and it participates in (does not get skipped by)
  aggregations. `skipna` skips nulls (validity 0), not NaNs. See `dtypes.md`.
- **`alloc` OOM** returns `0` (no trap); callers check.

## 8. Reference kernel — `add_f64` (implement with zero questions)

This is the worked example every elementwise kernel follows.

- **Symbol / family:** `add_f64`; elementwise family (`wasm/**/elementwise*`,
  `src/kernels/elementwise/`).
- **Rust signature:**
  ```rust
  #[no_mangle]
  pub unsafe extern "C" fn add_f64(
      a_ptr: *const f64,   // i32 byte offset, 16-byte aligned, len f64s
      b_ptr: *const f64,   // i32 byte offset, 16-byte aligned, len f64s
      out_ptr: *mut f64,   // i32 byte offset, 16-byte aligned, len f64s (caller-allocated)
      len: u32,            // element count
  ) { /* out[i] = a[i] + b[i] for i in 0..len */ }
  ```
- **Wasm signature:** `(i32 a_ptr, i32 b_ptr, i32 out_ptr, i32 len) -> ()`.
- **Semantics:** for `i in 0..len`, `out[i] = a[i] + b[i]`. `len` is an element count;
  `out` must be pre-allocated with `len * 8` bytes. No allocation, no `memory.grow`, no
  trap on valid inputs. `len == 0` is a valid no-op.
- **Validity / nulls:** `add_f64` is **data-only**. Null propagation for a binary op is
  performed separately by `validity_and(a_valid_ptr, b_valid_ptr, out_valid_ptr, len)`
  (operates on `ceil(len/8)` bytes; `validity_ptr == 0` means all-valid). The Phase-3
  expression compiler emits the `validity_and` alongside the arithmetic (and fuses it).
  Computing `a[i]+b[i]` on a lane that will be marked null is harmless and cheaper than
  branching.
- **SIMD variant** (`#[cfg(target_feature="simd128")]`): process 2 f64 per iteration
  with `f64x2` load/add/store, scalar tail for the odd element. Scalar variant is the
  plain loop. Both must produce identical results.
- **Determinism:** elementwise, so trivially identical across builds.

Unary elementwise (e.g. `neg_f64`) drops `b_ptr`; comparison elementwise emits a mask
(`gt_f64_mask(a_ptr, scalar, out_mask_ptr, len)`, §4.2); `_scalar` variants take an
immediate `f64`/`i32` operand instead of `b_ptr`.

## 9. Concrete export list (what Phase 1 & Phase 2 must implement)

Signatures use wasm value types. `vp` = `validity_ptr` (`0`=all-valid). `dt` ∈
{`f64,f32,i32,u32`} unless noted; `bool` data is `u8`.

### Phase 1 — memory core (`contracts/memory.d.ts` companion)
| Export | Signature | Notes |
|---|---|---|
| `memory` | (exported `WebAssembly.Memory`) | §2 |
| `alloc` | `(i32 size)->i32` | 16-byte aligned; `0` on OOM (§3) |
| `free` | `(i32 ptr)->()` | §3 |
| `realloc` | `(i32 ptr,i32 new_size)->i32` | §3 |
| `mem_generation` | `()->i32` | changes on every grow (§2) |

### Phase 2 — kernels (per agent; every kernel ships scalar + SIMD)

**Agent A — elementwise** (`elementwise*`):
| Export | Signature |
|---|---|
| `add_dt` `sub_dt` `mul_dt` `div_dt` `mod_dt` | `(i32 a,i32 b,i32 out,i32 len)->()` |
| `add_dt_scalar` … | `(i32 a, <dt> s, i32 out, i32 len)->()` |
| `neg_dt` | `(i32 a,i32 out,i32 len)->()` |
| `gt_dt_mask` `ge_dt_mask` `lt_dt_mask` `le_dt_mask` `eq_dt_mask` `ne_dt_mask` | `(i32 a,i32 b,i32 out_mask,i32 len)->()` |
| `gt_dt_scalar_mask` … | `(i32 a,<dt> s,i32 out_mask,i32 len)->()` |
| `and_kleene` `or_kleene` | `(i32 a,i32 a_vp,i32 b,i32 b_vp,i32 out,i32 out_vp,i32 len)->()` (Kleene, `dtypes.md`) |
| `not_bool` | `(i32 a,i32 a_vp,i32 out,i32 out_vp,i32 len)->()` |
| `validity_and` `validity_or` | `(i32 a_vp,i32 b_vp,i32 out_vp,i32 len)->()` |
| `cast_<from>_<to>` | `(i32 in,i32 in_vp,i32 out,i32 out_vp,i32 len)->()` (matrix in `dtypes.md`) |
| `fill_null_dt` | `(i32 in,i32 vp,<dt> fill,i32 out,i32 len)->()` (out all-valid) |
| `is_null` | `(i32 vp,i32 out_bool,i32 len)->()` (u8 bool result; no nulls) |
| `expand_mask_bool` | `(i32 mask,i32 out_u8,i32 len)->()` (1-bit mask → u8 bool column) |

**Agent B — reductions** (`reduce*`), null-aware, `skipna` (`dtypes.md`):
| Export | Signature |
|---|---|
| `sum_dt_null` | `(i32 data,i32 vp,i32 len)-> <dt/ f64 for ints>` |
| `mean_dt_null` | `(i32 data,i32 vp,i32 len)->f64` |
| `min_dt_null` `max_dt_null` | `(i32 data,i32 vp,i32 len)-> <dt>` (null→see dtypes) |
| `count_null` | `(i32 vp,i32 len)->i32` (non-null count) |
| `std_dt_null` `var_dt_null` | `(i32 data,i32 vp,i32 len)->f64` (ddof=1) |
| `nunique_dt_null` | `(i32 data,i32 vp,i32 len)->i32` |
| `first_dt_null` `last_dt_null` | `(i32 data,i32 vp,i32 len,i32 out_valid)-> <dt>` (out_valid = 0 if all null) |

**Kernel-level results for empty / all-null inputs** (the API layer maps these to
`null` per `dtypes.md` §4.3 using `count_null` — kernels just return deterministically):
`sum_*` → `0` (additive identity); `mean/std/var_*` → `NaN`; `min/max_f64/f32` → `NaN`;
`min/max_i32/u32` → `0` (callers must consult `count_null`); `std/var` with fewer than
2 non-null values → `NaN`. `nunique`: `NaN` counts as one distinct value (`dtypes.md`
§4.6).

**Prescribed accumulation order (both builds MUST implement it identically):**
`sum`/`mean` over `f64` use **2 striped accumulators** (element `i` → accumulator
`i & 1`), combined `acc0 + acc1` at the end; `f32` uses **4 striped accumulators**
combined left-to-right. Null lanes contribute the additive identity. `std`/`var` are
**two-pass** (striped mean, then striped sum of squared deviations, ddof=1). The scalar
build simulates the same striping so both binaries return bit-identical results.
Integer sums are order-insensitive (wrapping); min/max/count/nunique/first/last are
order-insensitive by nature.

**Agent C — selection** (`select*`):
| Export | Signature |
|---|---|
| `filter_dt` | `(i32 data,i32 mask,i32 out,i32 len)->i32` (compacts values where mask bit=1; returns out count) |
| `filter_indices` | `(i32 mask,i32 out_idx,i32 len)->i32` (mask → `i32` row indices; returns count) |
| `gather_dt` | `(i32 data,i32 idx,i32 idx_len,i32 out)->()` (take by index; `out[k]=data[idx[k]]`) |
| `gather_validity` | `(i32 vp,i32 idx,i32 idx_len,i32 out_vp)->()` |
| `argsort_dt` | `(i32 data,i32 vp,i32 inout_perm,i32 len,i32 desc,i32 scratch_ptr)->()` (stable; `inout_perm` holds a caller-initialized permutation — identity for single-key — and is stably reordered by `data[perm[i]]`; multi-key = thread it last-key-first; ordering per `dtypes.md` §4.6; **v1.2:** `scratch_ptr` = caller-allocated, 16-byte-aligned `i32[len]` merge scratch — the JS dispatch layer allocates/frees it around the call) |
| `topk_dt` | `(i32 data,i32 vp,i32 k,i32 out_idx,i32 len,i32 largest)->i32` (returns count written; semantics per `dtypes.md` §4.6) |

**v1.2 amendments (orchestrator, post-P2 bench triage).** (1) `argsort_dt` gains
`scratch_ptr` above: the no-alloc rotation-merge sort was correct but missed the §5
bench gate by orders of magnitude; a caller-provided `i32[len]` scratch restores
O(n log n) stable merge (or radix) without touching the §5.4 no-alloc rule. (2)
`filter_indices` stays exported but the **JS dispatch layer uses a JS implementation**:
investigation per §5 found V8's JIT beats wasm on this bitmap→index scatter loop
(0.59×; `Math.clz32` lowers to native ctz and JS avoids the boundary). Callers go
through `src/kernels/select/` and get the fast path automatically.

**Agent D — relational / hash** (`hash*`, ADR-005) — v1.1: signatures finalized by
orchestrator; the hash *function* is Agent D's choice (documented in code), but null
rows MUST hash to the reserved constant `H_NULL = 0x9e37_79b9_7f4a_7c15` so nulls form
one group in `group_build` (`dtypes.md` §4.5: null keys group together, but never
match in joins — join kernels use the validity bitmaps to exclude them):
| Export | Signature |
|---|---|
| `hash_dt` | `(i32 data,i32 vp,i32 out_hash,i32 len)->()` (64-bit hashes → `i64[len]`; null rows → `H_NULL`) |
| `hash_combine` | `(i32 acc_hash,i32 add_hash,i32 len)->()` (multi-key; in-place into `acc_hash`) |
| `group_build` | `(i32 hash_ptr,i32 len,i32 ht_ptr,i32 ht_cap,i32 out_group_ids)->i32` |
| `join_hash_inner` | `(i32 lh_ptr,i32 l_vp,i32 l_len,i32 rh_ptr,i32 r_vp,i32 r_len,i32 ht_ptr,i32 ht_cap,i32 out_l_idx,i32 out_r_idx,i32 out_cap)->i32` |
| `join_hash_left` | same as `join_hash_inner` |

- **Caller-provided hash-table scratch (keeps the §5.4 no-alloc rule):** `ht_ptr` is a
  caller-allocated open-addressing table of `ht_cap` slots (`ht_cap` a power of two;
  slot layout is Agent D's choice, ≤ 16 bytes/slot, zero-initialized by the caller).
  If the table cannot hold the build side, the kernel returns **`-1`** and the caller
  doubles `ht_cap` and re-calls (JS layer starts at `next_pow2(2 * build_len)` clamped
  to a sane minimum).
- **`group_build`:** assigns dense group ids in **order of first occurrence** (row 0's
  key = group 0 — deterministic across builds; hash-order grouping is non-conformant).
  Returns `group_count` (≥ 0) or `-1` (grow ht). `out_group_ids` is `i32[len]`.
  Grouping compares 64-bit hashes only (ADR-005; collision risk accepted & documented).
- **`join_*`:** builds on the **right** side, probes left rows in order. Returns the
  total pair count `n` (or `-1` to grow ht); writes at most `out_cap` pairs — if
  `n > out_cap` the caller re-allocates both out arrays to `n` and re-calls. Output
  order: probe (left-row) order; duplicate right matches in build order.
  `join_hash_left` emits `(l_idx, -1)` for unmatched/null-key left rows (the frame
  layer turns `-1` into null gathers). Null-validity rows never match (§ above).
- **`unify_dict`: dropped from the v1 wasm ABI.** Phase-1's JS-side dictionary
  unification (`src/memory/dictionary.ts`) is the v1 path; revisit with a benchmark if
  profiling shows it hot (would need an ADR note, not a reversal).

---

**Definition of "done" for a kernel agent:** implement the listed exports in both builds
following §5–§8, pass the orchestrator's conformance fixtures (including null/NaN/empty
cases), meet the §5 per-kernel bench gate, and touch no files outside the assigned path.

---

# v2 additions — i64 exports + temporal reuse (orchestrator-owned)

**Status:** v2, authoritative. Extends §1–§9 for ADR-009 (i64) and ADR-010 (temporals).
All §5 calling conventions, §7 trap policy, and §8 reference-kernel patterns are
**unchanged** and apply to every export below. Type/cast/null semantics: `dtypes.md`
v2 §6–§11.

## 10. i64 — the new kernel exports (`dt` = `i64` throughout)

Only **i64** adds wasm exports; temporals add **none** (§11). Every i64 export ships in
**both** builds with identical behavior (§1); the SIMD build uses `i64x2` only where §10.5
allows. `vp` = validity_ptr (`0` = all-valid, §4.1). i64 data buffers are 8-byte elements
at 16-byte-aligned base pointers.

### 10.1 BigInt64 boundary crossing (state it explicitly)

A wasm `i64` **parameter or return** surfaces in JS as a **`BigInt`** (the WebAssembly
JS-API BigInt↔i64 marshalling; Node ≥ 18 and evergreen browsers). So:

- **Scalar-operand kernels** (`add_i64_scalar`, `gt_i64_scalar_mask`, `fill_null_i64`)
  take an `i64` immediate → the JS dispatch passes a **`bigint`**, not a `number`.
- **i64 reductions that return an i64** (`sum_i64_null`, `min_i64_null`, `max_i64_null`,
  `first_i64_null`, `last_i64_null`) return a wasm `i64` → the JS layer receives a
  **`bigint`**. Reductions returning `f64`/`i32` (`mean/std/var_i64_null`,
  `nunique_i64_null`) are unaffected (`number`).
- **Bulk column data** never crosses as scalars — it stays in wasm memory behind a
  `BigInt64Array` `viewOf` (ADR-001); only the scalars above are `bigint`-boxed.

### 10.2 Agent A — elementwise (`elementwise*`)

| Export | Signature |
|---|---|
| `add_i64` `sub_i64` `mul_i64` `div_i64` `mod_i64` | `(i32 a,i32 b,i32 out,i32 len)->()` |
| `add_i64_scalar` `sub_i64_scalar` `mul_i64_scalar` `div_i64_scalar` `mod_i64_scalar` | `(i32 a, i64 s, i32 out, i32 len)->()` — **`s` is BigInt in JS** |
| `neg_i64` | `(i32 a,i32 out,i32 len)->()` |
| `gt_i64_mask` `ge_i64_mask` `lt_i64_mask` `le_i64_mask` `eq_i64_mask` `ne_i64_mask` | `(i32 a,i32 b,i32 out_mask,i32 len)->()` |
| `gt_i64_scalar_mask` `ge_i64_scalar_mask` `lt_i64_scalar_mask` `le_i64_scalar_mask` `eq_i64_scalar_mask` `ne_i64_scalar_mask` | `(i32 a, i64 s, i32 out_mask, i32 len)->()` — **`s` is BigInt** |
| `cast_f64_i64` `cast_f32_i64` `cast_i32_i64` `cast_u32_i64` `cast_bool_i64` | `(i32 in,i32 in_vp,i32 out,i32 out_vp,i32 len)->()` |
| `cast_i64_f64` `cast_i64_f32` `cast_i64_i32` `cast_i64_u32` `cast_i64_bool` | `(i32 in,i32 in_vp,i32 out,i32 out_vp,i32 len)->()` |
| `fill_null_i64` | `(i32 in,i32 vp, i64 fill, i32 out,i32 len)->()` — **`fill` is BigInt**; out all-valid |

- `add/sub/mul` **wrap** mod 2^64 (`dtypes.md` §6/§8). `div_i64`/`mod_i64` **truncate**
  toward zero; **zero divisor → clear out validity (null), no trap** (§7, `dtypes.md`
  §3.2); the JS layer supplies an `out_vp` for these two exactly as for `div_i32`.
- `cast_*` semantics per `dtypes.md` §7.1 (⚠ `f64/f32→i64` range/NaN → null; `i64→f64/f32`
  round-not-null; `i64→i32/u32` wrap-truncate). `is_null`, `expand_mask_bool`,
  `and/or_kleene`, `not_bool`, `validity_and/or` are **dtype-agnostic / bool-only** — no
  i64 variants.

### 10.3 Agent B — reductions (`reduce*`), null-aware, `skipna`

| Export | Signature |
|---|---|
| `sum_i64_null` | `(i32 data,i32 vp,i32 len)-> i64` — **BigInt return**; wrapping |
| `mean_i64_null` | `(i32 data,i32 vp,i32 len)-> f64` |
| `min_i64_null` `max_i64_null` | `(i32 data,i32 vp,i32 len)-> i64` — **BigInt return** |
| `std_i64_null` `var_i64_null` | `(i32 data,i32 vp,i32 len)-> f64` (ddof=1) |
| `nunique_i64_null` | `(i32 data,i32 vp,i32 len)-> i32` |
| `first_i64_null` `last_i64_null` | `(i32 data,i32 vp,i32 len,i32 out_valid)-> i64` — **BigInt return**; `out_valid=0` if all null |

`count_null` is validity-only (§9 Agent B) and is **reused** — no `count_i64`.

### 10.4 Agent C — selection (`select*`) & Agent D — hash (`hash*`)

| Export | Signature |
|---|---|
| `filter_i64` | `(i32 data,i32 mask,i32 out,i32 len)->i32` (compacts where mask bit=1; returns count) |
| `gather_i64` | `(i32 data,i32 idx,i32 idx_len,i32 out)->()` |
| `argsort_i64` | `(i32 data,i32 vp,i32 inout_perm,i32 len,i32 desc,i32 scratch_ptr)->()` (v1.2 `scratch_ptr` = caller `i32[len]`; stable; order per `dtypes.md` §4.6 — nulls last, no NaN) |
| `topk_i64` | `(i32 data,i32 vp,i32 k,i32 out_idx,i32 len,i32 largest)->i32` |
| `hash_i64` | `(i32 data,i32 vp,i32 out_hash,i32 len)->()` (64-bit hashes → `i64[len]`; **splitmix64 direct** on the value; null rows → `H_NULL`) |

`gather_validity`, `filter_indices` (JS dispatch, v1.2), `hash_combine`, `group_build`,
`join_hash_inner`, `join_hash_left` are **dtype-agnostic** (they operate on validity
bitmaps, masks, indices, or 64-bit hash values) and are **reused unchanged** — no i64
variants. `nunique_i64_null` may keep the documented O(n²)-beyond-scratch ceiling
(`status.md` parking lot).

### 10.5 SIMD policy for i64 (both builds, identical results)

- `add_i64` `sub_i64` `neg_i64` and all six `*_i64_mask` comparisons use **`i64x2`**
  (2 lanes/iter, scalar tail) in the SIMD build (`i64x2.add/sub/neg`, `i64x2.eq/ne/
  lt_s/gt_s/le_s/ge_s`). i64 is signed; use the `_s` compares.
- `mul_i64` `div_i64` `mod_i64` are **scalar-only in BOTH builds**: SIMD128 has no vector
  integer division/remainder, and `i64x2.mul` is emulated (no throughput win) — per the
  `hash.rs` precedent (ADR-009 §Decision/SIMD).

### 10.6 Scratch & accumulation rules for i64 reductions (both builds MUST agree)

- **`sum_i64_null` is order-insensitive.** Wrapping i64 add (mod 2^64) is associative and
  commutative, so the SIMD `i64x2` pairwise sum (combine the two lanes with a final
  wrapping add) and the scalar single-accumulator sum produce **bit-identical** results.
  No striped-accumulator determinism dance is needed (unlike f64/f32 in §9). Null lanes
  contribute `0`.
- **`mean_i64` / `std_i64` / `var_i64`** convert each i64 → f64 first (rounding beyond
  ±2^53, `dtypes.md` §6) and then run the **same f64 accumulation strategy as §9's float
  reductions** (2 striped f64 accumulators for the sum; two-pass, ddof=1, for std/var) so
  scalar and SIMD agree bitwise. Any bounded scratch is stack-local (§5.4); no `alloc`.
- **`min/max/first/last/count/nunique_i64`** are order-insensitive by nature.

### 10.7 Kernel-level results for empty / all-null i64 inputs (the §9 table, i64 row)

The API layer maps these to `null` via `count_null` (`dtypes.md` §11); kernels return
deterministically:

| kernel | empty / all-null result |
|---|---|
| `sum_i64_null` | **`0` (`0n`)** — additive identity |
| `mean_i64_null` `std_i64_null` `var_i64_null` | **`NaN`** (f64) |
| `min_i64_null` `max_i64_null` | **`0` (`0n`)** — callers **must** consult `count_null` (matches the `min/max_i32/u32` integer rule; there is no NaN sentinel for integers) |
| `first_i64_null` `last_i64_null` | value `0n`, **`out_valid = 0`** |
| `nunique_i64_null` | **`0`** |
| `count_null` | `0` (non-null count) |

`std/var_i64` with fewer than 2 non-null values → `NaN`, as in §9.

## 11. Temporals reuse existing exports — NO temporal-specific wasm exports

**There are ZERO temporal wasm exports.** `date32` and `timestamp` are logical dtypes
that dispatch to a **physical kernel token** (ADR-010 registry; `DTypeInfo.wasm`):

| logical dtype | kernel token | value-ops reused |
|---|---|---|
| `date32` | **`i32`** | `eq/ne/gt/ge/lt/le_i32_mask` (+ `_scalar_mask`), `argsort_i32`, `topk_i32`, `filter_i32`, `gather_i32`, `min/max/first/last_i32_null`, `sum/mean/std/var/nunique_i32_null` (where meaningful), `hash_i32`, `group_build`, `join_hash_*`, `cast_i32_*`, `add_i32`/`sub_i32` (restricted, §below) |
| `timestamp` | **`i64`** | the §10 `*_i64` equivalents of the above |

- **Compare / sort / group / join / min / max / first / last / filter / gather** on a
  temporal column call the token's kernel **unchanged**; the frame/expr layer relabels
  the result dtype back to the logical type.
- **Arithmetic is enforced in the JS expr/frame layer**, not in wasm: the restricted
  temporal algebra (`dtypes.md` §9) lowers to the token's `add/sub` kernels
  (`timestamp ± ms` → `add_i64`/`sub_i64`; `date32 ± days` → `add_i32`/`sub_i32`;
  `timestamp − timestamp` → `sub_i64`; `date32 − date32` → `sub_i32`), then relabels the
  result. Disallowed temporal ops never reach a kernel — the compiler raises the dtype
  error.
- **Scale-casts** (`dtypes.md` §7.2): `date32 → timestamp` lowers to `cast_i32_i64` +
  `mul_i64_scalar(86_400_000)` (existing kernels); `timestamp → date32` is a **JS-side**
  BigInt floor-div (no floor-div kernel exists; `div_i64` truncates). Reinterpret casts
  (`date32↔i32`, `timestamp↔i64`) emit **no** kernel (a pure relabel).
- **`dt` accessors are JS-side** (civil-from-days integer math; tz-aware via cached
  `Intl.DateTimeFormat`) — **no kernel** (ADR-010).

Consequence: **no wasm binary grows for temporals**; the §1 75 KB-gzipped size budget is
affected only by the i64 kernels of §10.

---

## 12. v2.2 amendment — census-perf workstream (orchestrator, 2026-07-03)

**Motivation:** a real 85K-row workload joining on a ~unique `utf8` key measured 4×
slower than Arquero end-to-end (bead `dataframe-4w8`); the JS-side dictionary
unification deferred in §9 ("revisit with a benchmark if profiling shows it hot") is
hot, and dictionary ingest of high-cardinality strings dominates load time.

**New export (hash family, both builds, identical results):**
| Export | Signature | Semantics |
|---|---|---|
| `hash_utf8_dict` | `(i32 offsets_ptr, i32 bytes_ptr, i32 dict_count, i32 out_hash_ptr) -> ()` | For each dictionary slot `k in 0..dict_count`, hash the raw UTF-8 bytes `bytes[offsets[k]..offsets[k+1])` to a 64-bit hash written to `out_hash[k]` (`i64[dict_count]`). Deterministic, identical across builds, same hash-quality class as `hash_dt` (splitmix64-finalized byte mixing is fine; document the exact scheme in code). Empty string hashes to a fixed non-zero value. `H_NULL` is NOT involved (row nullness lives in row validity, not the dictionary). |

**Join path change (no observable semantics change):** utf8 equi-joins no longer unify
dictionaries. Each side computes `hash_utf8_dict` over its own dictionary once, rows
gather their key hash via `gather_i64(dict_hashes, indices)`, and `join_hash_inner/left`
consume those hashes unchanged (row validity still excludes nulls per §9). Output key
columns reuse the **left** dictionary (inner: all emitted keys exist on the left; left
join: identical). 64-bit collision risk is unchanged in class from ADR-005 and remains
accepted+documented. Dictionary unification (`src/memory/dictionary.ts`) remains for
concat-style ops; joins simply stop calling it.

**Ingest note (within ADR-002):** `fromArrow`/dictionary build may dedup on raw UTF-8
bytes (e.g. via `hash_utf8_dict` + byte-compare on collision) without materializing JS
strings; per-slot decode stays lazy/memoized (ADR-002 unchanged). Already-dictionary-
encoded Arrow input passes its dictionary through without a rebuild.
