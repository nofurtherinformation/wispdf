# Phase-2 Kernel Conformance Fixtures

**Status:** orchestrator-owned, read-only to kernel agents.

These JSON fixtures are the acceptance tests for every Phase-2 kernel export (ABI §9).
A kernel agent passes conformance when every non-BLOCKED case in the relevant fixture file
produces byte-identical output to the `expected` values below.  BLOCKED cases mark spots
where the ABI is silent; the orchestrator will fill them in before or during verification.

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
  "expected": { ... }             // absent when "blocked": true
}
```

A BLOCKED case (ABI gap; do NOT implement around):

```jsonc
{
  "export":  "<kernel_name>",
  "name":    "BLOCKED__<reason>",
  "blocked": true,
  "blocked_reason": "...",
  "inputs":  { ... }
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

---

## 4. Accumulation-order-sensitive cases

Floating-point reductions are not associative.  Cases tagged with
`"accumulation_order": "left_to_right"` specify the expected value for naive sequential
summation (accumulate left-to-right starting from the identity element).  An
implementation that documents a different accumulation strategy (e.g. pairwise) may
produce a different result; the conformance verifier accepts this ONLY IF the
implementation's stated strategy matches its actual output AND scalar and SIMD produce
the same value.

---

## 5. Coverage requirements (kernel agents must not delete or weaken any case)

Every non-BLOCKED export has, at minimum:

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

---

## 6. BLOCKED cases (ABI gaps to resolve before verification)

The following semantics are not fully specified in `dtypes.md` or `wasm-abi.md`.
The orchestrator must resolve them before these cases can be enabled.

| kernel(s)                                       | gap |
|-------------------------------------------------|-----|
| `min_dt_null`, `max_dt_null`                    | Null return when all-null: ABI §9 signature has no `out_valid` pointer. For integer dtypes there is no sentinel value for null. dtypes.md says result is null; signature is `(data,vp,len) -> <dt>` with no way to signal null. Does an `out_valid i32 ptr` need to be added? |
| `mean_dt_null`, `std_dt_null`, `var_dt_null`    | Same issue: null result (all-null for mean, <2 non-null for std/var) has no signaling mechanism in the ABI signature. |
| `argsort_dt`                                    | Null ordering: dtypes.md §4 does not state whether null values sort before or after non-null values. |
| `topk_dt`                                       | Sort direction: ABI §9 says "top-k" but does not specify largest vs smallest. |
| `join_hash_inner`, `join_hash_left`             | Out-param shape explicitly deferred to Agent D's brief (ABI §9 note). |
| `unify_dict`                                    | Out-param shape explicitly deferred to Agent D's brief (ABI §9 note). |
