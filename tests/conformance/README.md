# Phase-2 Kernel Conformance Fixtures

**Status:** orchestrator-owned, read-only to kernel agents.

These JSON fixtures are the acceptance tests for every Phase-2 kernel export (ABI §9).
A kernel agent passes conformance when every case in the relevant fixture file
produces byte-identical output to the `expected` values below.  All previously-BLOCKED
cases have been filled per contracts v1.1 (commit 954efc8 + this update); the BLOCKED gap
table (§6) is now empty.

---

## 1. Fixture schema

Each fixture file has the shape:

```jsonc
{
  "family": "<elementwise|reductions|selection|relational>",
  "cases": [ <case>, ... ]
}
```

A case:

```jsonc
{
  "export":  "<kernel_name>",      // exact wasm export symbol
  "name":    "<snake_case_id>",    // unique within the file
  "note":    "optional prose",
  "inputs":  { ... },
  "expected": { ... }
}
```

---

## 2. Buffer encoding rules

### 2.1 Data arrays

| dtype      | JSON representation                                  |
|------------|------------------------------------------------------|
| `f64`      | JSON number; specials: `"NaN"`, `"Infinity"`, `"-Infinity"` |
| `f32`      | Same as f64; runner packs into Float32Array          |
| `i32`      | JSON integer in `[-2147483648, 2147483647]`          |
| `u32`      | JSON integer in `[0, 4294967295]`                    |
| `bool`     | JSON integer 0 (false) or 1 (true); u8 storage       |
| `i64` hash | Decimal string e.g. `"1"`, `"9223372036854775807"`  |

### 2.2 Validity bitmaps (`*_vp` keys)

A validity bitmap is expressed as a **flat array of 0/1 integers**, one per element:
`1 = valid`, `0 = null`.  Element count equals the data array length.

The runner packs these into Arrow-LSB bytes before passing the pointer:
`bitmap[i >> 3] |= value << (i & 7)`.  Padding bits beyond `len` in the last byte
are written as `0`.

**All-valid fast path:** omit the `*_vp` key entirely (do NOT include it as `[]`).
The runner passes `validity_ptr = 0`, which the kernel must treat as all-valid (ABI §4.1).
Every null-aware kernel must have at least one case that omits the validity key.

**All-null:** include `*_vp` as an array of all `0`s.

### 2.3 Mask arrays (`*_mask` keys)

Comparison kernels produce a 1-bit-per-element mask in the same Arrow-LSB layout as
validity bitmaps.  In fixtures, expressed as a flat `0/1` integer array (one per element).
The runner packs them the same way as validity bitmaps.

### 2.4 Scalar parameters

A scalar wasm parameter (e.g. `<dt> s` in `add_dt_scalar`) is a single JSON value
following the dtype encoding above.  It appears as `"s": <value>` in `inputs`.

### 2.5 Unspecified output slots

A JSON `null` in an `expected` data array means the slot is written by the kernel
but its value is implementation-defined (ABI §7: "unspecified but written, typically 0").
The runner skips equality checks for `null` positions.

### 2.6 Property-test cases

Some relational cases carry a `"property"` field instead of (or alongside) `expected`.
Recognized properties:

| property | verification rule |
|----------|------------------|
| `"equal_inputs_equal_hashes"` | positions that share a data value must produce identical i64 hashes; null positions must all produce the same i64 hash as each other |
| `"group_partition"` | `expected.group_count` must equal the number of distinct groups; `expected.partitions` lists which input indices share a group ID. Comparison is **set-of-sets**: hash-table group order and element order within each group are implementation-defined — `[[0,2],[1,4],[3]]` and `[[3],[1,4],[0,2]]` are equivalent. |

---

## 3. Mapping a fixture case to a kernel call

1. **Allocate** input and output buffers in wasm linear memory via `alloc`.
2. **Copy** data arrays into the appropriate TypedArray view (Float64Array for f64,
   Float32Array for f32, Int32Array for i32/bool/u8, Uint32Array for u32, BigInt64Array
   for i64 hashes).
3. **Pack** `*_vp` and `*_mask` arrays into bytes (Arrow LSB, §2.2).
   If a `*_vp` key is absent, pass `0` as the validity pointer.
4. **Call** the exported function with the buffer pointers and `len` (element count of
   the primary data array).
5. **Compare** the output buffer values to `expected`, skipping JSON-`null` slots.
   For floats use bit-pattern comparison (`Object.is`) so that NaN===NaN and +0!=-0 are
   respected; for integers use exact equality.
6. **Free** all allocated buffers.

For **reductions** returning a scalar: the return value is compared to `expected.result`.
String specials `"NaN"/"Infinity"/"-Infinity"` are matched against the IEEE bit pattern.

For **first/last**: `expected.out_valid` (0 or 1) is checked first; if 0, the returned
scalar is not checked (implementation-defined when result is null).

For **group_build** with `property: "group_partition"`: read `*out_group_count_ptr`;
check it equals `expected.group_count`; then verify the partition structure of
`out_group_ids` matches `expected.partitions`.  Partition comparison is
**unordered at both levels**: group order and index order within each group are
implementation-defined (hash-table order). Sort each group's index list and sort
the groups themselves before comparing.

### Join cases (`join_hash_inner`, `join_hash_left`)

The fixture supplies the pre-computed hash arrays (`lh`, `rh` as i64 decimal string
arrays) and validity bitmaps (`l_vp`, `r_vp`).  The runner is responsible for:

- Sizing `ht_ptr`/`ht_cap` (start at `next_pow2(2 * build_len)`, minimum 4 slots,
  zero-initialized).
- Sizing `out_l_idx`/`out_r_idx`/`out_cap` (start at `build_len + probe_len`).
- Implementing the **-1 grow-and-retry protocol**: if the kernel returns `-1`, double
  `ht_cap`, re-zero and re-call; if it returns `n > out_cap`, reallocate out arrays
  to `n` and re-call.

`expected.pairs` is an ordered list of `[l_idx, r_idx]` pairs in probe (left-row) order;
duplicate right matches are in build (right-row) order.  The runner verifies the full
pair list in order.

`join_hash_left` emits `[l_idx, -1]` for unmatched left rows **and** for null-validity
left rows (vp=0).  The frame layer maps `-1` right indices to null-gather operations.

`H_NULL = 0x9e3779b97f4a7c15` (signed i64: `-7046029254386353131`) is the reserved
null-row hash emitted by `hash_dt` for null-validity rows.  Join fixtures use `"0"` as
a placeholder hash value for null rows since the validity bit (vp=0) governs matching,
not the hash value.

---

## 4. Accumulation order — PRESCRIBED (not implementation-chosen)

Floating-point reductions are not associative. The ABI prescribes specific accumulation
strategies so that scalar and SIMD builds return **bit-identical** results:

- **`f64` sum/mean:** 2 striped accumulators — element `i` goes to accumulator `i & 1`;
  combined as `acc0 + acc1` at the end.  Example: `[1e16, 1.0, -1e16]` →
  `acc0 = 1e16 + (-1e16) = 0.0`, `acc1 = 1.0` → result `1.0` (not `0.0`; the naive
  left-to-right result `0.0` does NOT conform).

- **`f32` sum/mean:** 4 striped accumulators — element `i` goes to accumulator `i & 3`;
  combined left-to-right in f32: `((acc0 + acc1) + acc2) + acc3`.  Example:
  `[1e8, 1.0, -1e8, 1.0, 1e8, 1.0, -1e8, 1.0]` → result `2.0` (not `1.0`; the naive
  left-to-right result `1.0` does NOT conform).

- **`std`/`var`:** two-pass — striped mean (same strategy per dtype), then striped sum
  of squared deviations, ddof=1.

Null lanes contribute the additive identity (0) to their accumulator slot (i.e. they
are skipped without breaking stripe alignment; the valid-element loop simply does not
add their contribution).

The conformance verifier uses **bit-pattern comparison** (`Object.is`) on float results.
An implementation that deviates from the prescribed strategy will fail the
`sum_f64_null__accumulation_order_sensitive` and `sum_f32_null__striping_sensitive` cases.

---

## 5. Coverage requirements (kernel agents must not delete or weaken any case)

Every non-blocked export has, at minimum:

- `len=0` (empty, a valid no-op)
- `len=1`
- `len` not a multiple of 8 (bitmap padding correctness for bitmapped ops)
- No-null fast path (validity key omitted → `validity_ptr=0`)  ← for null-aware kernels
- Mixed nulls (skipna behavior)                                 ← for null-aware kernels
- All-null                                                      ← for null-aware kernels
- NaN-as-value (valid NaN, not null) for f64/f32 kernels
- ±Infinity for f64/f32 arithmetic
- Zero-divisor for div/mod float (→ IEEE ±Inf/NaN)
- Zero-divisor for div/mod integer (→ kernel writes 0 or any value; null from caller)
- Full 9-row Kleene truth table for `and_kleene`/`or_kleene`
- Stable-sort case for `argsort` (equal keys preserve original order)
- Mask-padding case for `filter`/`gather` (len=9 so the second bitmap byte has padding)
- Multi-key threading case for `argsort` (two sequential calls threading `inout_perm`)
- desc-with-nulls case for `argsort` (nulls still last in descending)

---

## 6. BLOCKED cases (ABI gaps)

**All BLOCKED cases have been resolved.** The table below is empty.

| kernel(s) | gap |
|-----------|-----|
| *(none)*  |     |

### Resolution summary (contracts v1.1, commit 954efc8)

| Previously blocked | Resolution |
|--------------------|------------|
| `min/max_dt_null` empty/all-null | wasm-abi.md §9: float dtypes → `NaN`; i32/u32 → `0`. Callers use `count_null` to distinguish. |
| `mean/std/var_dt_null` empty/all-null | wasm-abi.md §9: → `NaN`. |
| `std/var_dt_null` fewer than 2 non-null | wasm-abi.md §9: → `NaN`. |
| `nunique_f64_null` NaN counts | dtypes.md §4.6: NaN counts as **one** distinct value. |
| `argsort_dt` null ordering | dtypes.md §4.6: nulls last both directions; NaN after +inf ascending / first descending; stability guaranteed. |
| `argsort_dt` signature | wasm-abi.md §9 C: `inout_perm` caller-initialized identity; `desc` param. |
| `topk_dt` direction / signature | dtypes.md §4.6 + wasm-abi.md §9 C: `largest` param; NaN participates as largest. |
| `join_hash_inner/left` out-param shape | wasm-abi.md §9 D: finalized; semantic fixtures added; runner owns ht/out sizing + retry. |
| `unify_dict` | **Dropped from v1 ABI.** JS-side unification (`src/memory/dictionary.ts`) is the v1 path. |
| `sum_f64_null` accumulation order | **Prescribed** 2-striped (not implementation-chosen); expected value updated to `1.0`. |
