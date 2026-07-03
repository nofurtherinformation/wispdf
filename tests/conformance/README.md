# Phase-2 / v2 Kernel Conformance Fixtures

**Status:** orchestrator-owned, read-only to kernel/v2 agents.

These JSON fixtures are the acceptance tests for every Phase-2 kernel export (ABI §9)
and the v2 i64/temporal additions (ABI §10–§11, ADR-009/010, bead dataframe-dh9.2).
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

| dtype        | JSON representation                                  |
|--------------|------------------------------------------------------|
| `f64`        | JSON number; specials: `"NaN"`, `"Infinity"`, `"-Infinity"` |
| `f32`        | Same as f64; runner packs into Float32Array          |
| `i32`        | JSON integer in `[-2147483648, 2147483647]`          |
| `u32`        | JSON integer in `[0, 4294967295]`                    |
| `bool`       | JSON integer 0 (false) or 1 (true); u8 storage       |
| **`i64`**    | **Decimal string** e.g. `"1"`, `"9223372036854775807"`, `"-9223372036854775808"`. ALL i64 data values — column data, scalar operands (`s`), reduction results — are decimal strings in the `i64.json` fixture. The runner converts these to `BigInt` via `BigInt(str)` and packs into `BigInt64Array`. This avoids JSON number precision loss for values above `Number.MAX_SAFE_INTEGER` (2^53−1). |
| `i64` hash   | Same decimal-string convention (these were already string-encoded in v1 relational fixtures). |

**BigInt boundary values used in fixtures:**

| constant    | decimal string | notes |
|-------------|---------------|-------|
| `INT64_MAX` | `"9223372036854775807"` | = 2^63 − 1 |
| `INT64_MIN` | `"-9223372036854775808"` | = −2^63 |
| `H_NULL`    | (implementation-defined constant `0x9e3779b97f4a7c15` as signed i64) | hash reserved for null rows |

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
For i64 scalar kernels (`add_i64_scalar`, `fill_null_i64`, etc.) the `s` / `fill`
value is a **decimal string** (same as i64 data), e.g. `"s": "86400000"`.
The runner passes it to wasm as a `BigInt`.

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

### 2.7 Frame-layer conventions shared by `i64.json` and `temporal.json`

Some cases in `i64.json` (the `i64_column_construction_errors` group) and all non-kernel cases in `temporal.json` carry a `layer` field:

| `layer` value | File(s) | Runner action |
|---|---|---|
| `"frame_error"` | `i64.json`, `temporal.json` | Invoke the JS frame/expr layer with the specified `op` and type information; assert that a descriptive Error is thrown. Match `error_pattern` as a case-sensitive substring of `error.message`. Never accept a silent wrong result. |

**i64.json `frame_error` — column construction errors:** the runner calls column construction for dtype `"i64"` with a plain `number[]` containing `input_values`. Cases cover: Number outside safe-integer range (|x| > 2^53-1), non-integer Number, and NaN Number (ADR-009 §Decision). The `input_values` array uses the same JSON encoding as the `in` array for float inputs (specials `"NaN"` etc.).

---

### 2.7a Temporal fixture conventions (`temporal.json`)

The `temporal.json` fixture covers the **JS registry+frame layer** (not new wasm
kernels — see wasm-abi.md §11: "ZERO temporal wasm exports").  Each case has a `layer`
field that tells the runner how to execute it:

| `layer` value | Runner action |
|---|---|
| `"kernel_reuse"` | Run the **physical kernel** identified by `dtype_physical` (`"i32"` or `"i64"`) with the specified `export`; verify the result matches `expected`. The logical temporal type is a label only. |
| `"frame"` | Invoke the JS frame/expr layer with the specified `cast` operation; verify output. |
| `"frame_error"` | Invoke the JS frame/expr layer with the specified `op` and dtype pair; verify that a **TypeError / RangeError / descriptive Error** is thrown (not a silent wrong answer). Match `error_pattern` as a substring of the error message. |
| `"frame_accessor"` | Invoke `.dt.<field>` accessor for the `dtype` and `tz`; verify each `expected` field value. |

**Temporal encoding in fixtures:**
- `date32` physical values appear as `i32` JSON integers (day counts).
- `timestamp` physical values appear as **decimal strings** (same i64 convention as `i64.json`).
- `out_ts_ms` / `data_ts_ms` are decimal strings; the runner converts via `BigInt`.

**tz-aware accessor ICU dependency:**  Cases with `"tz": "America/Chicago"` and similar
were computed with **Node v22.23.1 (ICU 78.2)**.  The DST transition on 2026-03-08 is
governed by the US Energy Policy Act of 2005.  If the runtime's ICU version or the legal
timezone rules differ, these cases may produce different field values.  The test runner
**must log the ICU version** and may skip tz-aware cases with a warning when mismatched.
The `icu_caveat` field on the fixture root documents the compute environment.

---

### 2.8 Parquet fixture conventions (`parquet.json`)

The `parquet.json` fixture covers the **`databonk/parquet` subpath** (ADR-011). Each case has a `layer` field:

| `layer` value | Runner action |
|---|---|
| `"parquet_roundtrip"` | Build a databonk DataFrame from `frame.columns`; call `writeParquet(df, write_opts)` to obtain Parquet bytes; call `readParquet(bytes)` to obtain a second DataFrame; assert every column in the second DataFrame matches `expected.columns` (dtype, nullability, and values). Additionally, validate the written bytes against the `parquet-wasm` oracle: use parquet-wasm to read the bytes and assert the same values. |
| `"parquet_error"` | If `"op": "write"` is present: call `writeParquet(df, write_opts)` on the supplied `frame` and assert it throws with `error_pattern` as a substring of the message. Otherwise: use the `parquet-wasm` oracle to generate a Parquet file matching `generate_with_oracle` (encoding, compression, data), then call `readParquet(bytes)` and assert it throws with `error_pattern`. The throw must be a descriptive Error — never a silent wrong result (ADR-011 §Consequences). |

**Parquet data encoding in fixtures** follows the same conventions as `i64.json`/`temporal.json`:
- `f64`/`f32`: JSON numbers (specials `"NaN"`, `"Infinity"`, `"-Infinity"`)
- `i32`/`u32`: JSON integers
- `i64`: decimal strings (e.g. `"9007199254740993"`)
- `bool`: 0 (false) or 1 (true)
- `utf8`: JSON strings or `null`
- `date32`: i32 JSON integer (day count since 1970-01-01)
- `timestamp`: decimal-string i64 (epoch ms since 1970-01-01 UTC)
- `null` in a data array: a null value for that row

The fixture root carries `"oracle": "parquet-wasm"` to identify the devDep oracle used for in-profile write verification and out-of-profile file generation.

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

## 5. Coverage requirements (kernel/v2 agents must not delete or weaken any case)

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

Additional v2 coverage requirements (beyond the v1 list above):

- For every i64 elementwise binary op: empty, len=1, len%2≠0 (scalar tail for i64x2 SIMD), and the wrapping case
- `div_i64`/`mod_i64` zero-divisor (kernel must not trap; writes 0 or unspecified)
- `div_i64(INT64_MIN, -1)` wraps to INT64_MIN (guarded, no trap, ADR-009)
- `neg_i64(INT64_MIN)` wraps to INT64_MIN (no trap)
- `cast_f64_i64`/`cast_f32_i64`: NaN→null, ±Infinity→null, |x|≥2^63→null (out-of-range), valid truncation
- `cast_i64_f64`: 2^53+1 = 9007199254740993 rounds to 9007199254740992.0 (round-to-even, row stays VALID); INT64_MAX rounds to 2^63 = 9223372036854775808.0 (VALID, not null)
- `cast_i64_i32`/`cast_i64_u32`: wrap-truncate (never null), INT64_MAX low-32 bits = -1/4294967295
- `sum_i64_null` wrapping: INT64_MAX+1+2 = -9223372036854775806
- `mean_i64_null`/`std_i64_null` precision case: [2^53, 2^53+2] → mean=9007199254740992.0 (not exact 9007199254740993), std=2.0 (not sqrt(2)) due to f64-first conversion
- `hash_i64`: equal-value property test; null rows hash to H_NULL property test
- All `argsort_i64`/`topk_i64` null-last cases (both directions)
- Temporal: all 5 floor-div cases for `timestamp→date32` (esp. -1 ms, -86400001 ms)
- Temporal: ISO weekday for full week spanning the epoch (days -4..+4)
- Temporal: tz-aware DST boundary (before/at 2026-03-08T08:00:00Z, America/Chicago)
- Temporal: all `frame_error` cases for disallowed arithmetic (§9 restriction table)

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

---

## 7. Fixture coverage table (v1 + v2)

| Fixture file | Family | Cases | Exports / semantics | Added |
|---|---|---|---|---|
| `elementwise.json` | elementwise | (see file) | Agent A: add/sub/mul/div/mod/neg, comparison masks, Kleene bool, cast, fill_null, is_null, expand_mask | v1 |
| `reductions.json` | reductions | (see file) | Agent B: sum/mean/min/max/std/var/nunique/first/last, null-aware, skipna | v1 |
| `selection.json` | selection | (see file) | Agent C: filter/gather/argsort/topk; Agent C+D: filter_indices | v1 |
| `relational.json` | relational | (see file) | Agent D: hash, group_build, join_hash_inner/left | v1 |
| **`i64.json`** | **i64** | **166** | **All 48 new i64 exports (wasm-abi.md §10): 5 binary + 5 scalar + neg + 6 cmp + 6 scalar-cmp + 10 casts + fill_null + 9 reductions + 5 selection + hash. Plus 4 frame_error cases for safe-int literal throw (ADR-009 §Decision). cast_f64_i64__range_null extended with prevDouble(2^63)=2^63-1024 to close boundary-off-by-ULP gap (verifier correction).** | **v2 (dh9.2)** |
| **`temporal.json`** | **temporal** | **62** | **date32/timestamp semantic layer: compare/sort/min/max via i32/i64 reuse; restricted arithmetic + null propagation; 9 cast cases incl. 5 negative floor-div; dt accessors (UTC + tz-aware DST boundary); 8 arithmetic-error cases; 4 hash property cases** | **v2 (dh9.2)** |
| **`parquet.json`** | **parquet** | **18** | **databonk/parquet subpath (ADR-011): 12 round-trip cases (all 9 supported dtypes individually — f64, f32, i32, u32, i64, bool, utf8, date32, timestamp-UTC, timestamp+tz — plus multi-dtype frame + Snappy); 6 out-of-profile error cases (gzip, zstd, INT96 timestamp, DECIMAL, LIST nested, invalid write codec). parquet-wasm devDep oracle for write validation and out-of-profile file generation.** | **v2 (dh9.2)** |
