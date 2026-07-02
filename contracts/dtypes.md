# Dtypes, casting & null semantics — v1 (orchestrator-owned contract)

**Status:** v1, authoritative. Companion to `contracts/wasm-abi.md`. Defines the v1 dtype
set, the **complete** explicit-cast matrix, the single implicit-conversion rule, and
null semantics (propagation, Kleene logic, `skipna` aggregations, `count` vs `size`).
Read-only to subagents. Source: spec §3 (locked), ADR-002.

---

## 1. Dtypes v1

| dtype | storage | bytes/value | scalar wasm type | validity | notes |
|---|---|---|---|---|---|
| `f64` | contiguous f64 | 8 | `f64` | bitmap | IEEE-754 double. Default float. |
| `f32` | contiguous f32 | 4 | `f32` | bitmap | IEEE-754 single. |
| `i32` | contiguous i32 | 4 | `i32` | bitmap | signed 32-bit. Default int. |
| `u32` | contiguous u32 | 4 | `i32` (unsigned semantics) | bitmap | unsigned 32-bit. |
| `bool` | contiguous **u8** (0/1) | 1 | `i32` (0/1) | bitmap | value storage is one byte per element (spec §3). Internal 1-bit comparison **masks** are a distinct layout — see `wasm-abi.md` §4.2/§4.4. |
| `utf8` | dict-encoded | i32 idx + i32 offsets + u8 bytes | — | bitmap (on the index buffer) | ADR-002 dictionary encoding. Kernels operate on `i32` indices. |

**Non-goals (v2, spec §0):** i64/BigInt, dates/timestamps/timezones, decimals. Not in
this matrix.

**Null representation (all dtypes):** a separate **validity bitmap** (Arrow LSB,
`1 = valid`, `wasm-abi.md` §4.1). Null is **never** encoded as a NaN/sentinel value at
the columnar/ABI level. A genuine `NaN` or `±inf` stored with validity bit `= 1` is a
**valid** value, not a null (see §4).

---

## 2. Explicit casts (`col('a').cast('<to>')`) — full matrix

Casting is **explicit only**. Row = source dtype, column = target dtype.
Legend: **=** exact / lossless · **≈** allowed, may lose precision · **⚠** allowed,
out-of-range/invalid inputs → **null** · **id** identity (no-op copy) · **✗** not in v1
(throws a helpful error naming both dtypes).

| from \ to | f64 | f32 | i32 | u32 | bool | utf8 |
|---|---|---|---|---|---|---|
| **f64**  | id | ≈ (round; overflow→±inf) | ⚠ trunc→0; out of `[-2³¹,2³¹-1]` or NaN → null | ⚠ trunc→0; out of `[0,2³²-1]`, neg, NaN → null | ⚠ x≠0→true, 0→false, NaN→null | ✗ |
| **f32**  | = (widen) | id | ⚠ (as f64→i32) | ⚠ (as f64→u32) | ⚠ (as f64→bool) | ✗ |
| **i32**  | = | ≈ (lose precision if \|x\|>2²⁴) | id | ⚠ neg → null; else same | x≠0→true, 0→false | ✗ |
| **u32**  | = | ≈ (lose precision if x>2²⁴) | ⚠ x>2³¹-1 → null; else same | id | x≠0→true, 0→false | ✗ |
| **bool** | = (t→1.0,f→0.0) | = | = (t→1,f→0) | = (t→1,f→0) | id | ✗ |
| **utf8** | ✗ | ✗ | ✗ | ✗ | ✗ | id |

Rules that apply to **every** cast:

- **Null propagates:** a null input row → null output row (validity bit copied), *in
  addition* to any range-failure nulls introduced by a **⚠** cast.
- **Float→int truncates toward zero** (not round/floor). `2.9→2`, `-2.9→-2`.
- **Range/invalid failure → null:** ⚠ casts set the output validity bit to `0` for
  inputs that overflow the target range, are `NaN`, or are negative into an unsigned
  target. They **never trap** (`wasm-abi.md` §7). Output data on a nulled slot is
  unspecified but written (typically `0`).
- **numeric↔`utf8` (✗):** number formatting / string parsing is **not** a v1 kernel
  cast (locale/format-fraught). String→number happens only via CSV typed inference
  (Phase 6); number→string display happens JS-side in `toString`/`describe`. A
  programmatic `cast` between `utf8` and a numeric dtype throws.
- **Kernel form:** `cast_<from>_<to>(in_ptr, in_vp, out_ptr, out_vp, len)`
  (`wasm-abi.md` §9). Identity casts are elided by the compiler (no kernel emitted).

---

## 3. Implicit conversion — the single rule

> **The ONLY implicit conversion in v1 is integer→float widening, and only in mixed-dtype
> arithmetic.** Everywhere else (comparisons, boolean ops, assignment, `withColumn`,
> `fillNull` value, join keys, concat/append) dtypes must match exactly, else a helpful
> error is raised naming both dtypes and the operation.

### 3.1 Arithmetic result-type lattice (`add sub mul`, and `div`/`mod` per §3.2)

| left ⊕ right | result | conversion |
|---|---|---|
| `f64` ⊕ `f64` | `f64` | none |
| `f32` ⊕ `f32` | `f32` | none |
| `i32` ⊕ `i32` | `i32` | none |
| `u32` ⊕ `u32` | `u32` | none |
| `i32` or `u32` ⊕ `f64` | `f64` | **int widened to f64** |
| `i32` or `u32` ⊕ `f32` | `f32` | **int widened to f32** (⚠ loses precision if \|x\|>2²⁴) |
| any **other** mixed pair (`i32`⊕`u32`, `f32`⊕`f64`, `bool`⊕anything, `utf8`⊕anything) | — | **error** — requires an explicit `cast` first |

Note: `f32`⊕`f64` is float→float, **not** int→float, so it is **not** implicit — it
errors and must be cast. `i32`⊕`u32` is int→int, also not covered — errors. This keeps
the rule to exactly one case.

### 3.2 Division & modulo gotcha (integer stays integer)

Because int⊕int→int (no implicit float), `div`/`mod` on two integer columns perform
**truncating integer division/remainder** and return an integer column — they do **not**
implicitly produce a float. For true (floating) division, `cast` an operand to `f64`
first. A **zero divisor** yields **null** (validity `0`), never a trap (`wasm-abi.md`
§7). Float `div`/`mod` follow IEEE-754 (`x/0 → ±inf`, `0/0 → NaN`).

- `neg` preserves dtype. Unary math keeps dtype. `bool` is not an arithmetic dtype
  (arithmetic on `bool` errors; cast to a numeric dtype first).

---

## 4. Null semantics (pandas-flavored, spec §3)

**Null ≠ NaN.** Null is validity-bitmap `0`. `NaN`/`±inf` with validity `1` are valid
values. `skipna` skips nulls, **not** NaNs — a valid `NaN` propagates through arithmetic
(→ NaN) and poisons `sum`/`mean` (→ NaN) exactly like numpy; `min`/`max` follow IEEE
(NaN is not selected unless all-NaN). This distinction is intentional and testable.

### 4.1 Propagation (arithmetic & comparison)

- **Unary op:** `out_valid[i] = in_valid[i]`; value computed where valid.
- **Binary op (`add sub mul div mod`, `gt ge lt le eq ne`):** `out_valid[i] =
  a_valid[i] & b_valid[i]` (null if **either** operand is null). Comparison masks: a null
  operand → that mask bit is `0` **and** the row is marked null via the accompanying
  validity; where a comparison feeds `filter`, a null predicate row is **dropped** (§4.5).
- Data on a null output slot is unspecified but written (branchless compute is fine).

### 4.2 Boolean `and`/`or` — Kleene three-valued logic

Values: `T` (valid,1), `F` (valid,0), `N` (null). `and_kleene`/`or_kleene`
(`wasm-abi.md` §9) implement:

| a | b | a AND b | a OR b |
|---|---|---|---|
| T | T | T | T |
| T | F | F | T |
| F | T | F | T |
| F | F | F | F |
| T | N | **N** | **T** |
| N | T | **N** | **T** |
| F | N | **F** | **N** |
| N | F | **F** | **N** |
| N | N | **N** | **N** |

Mnemonic: `AND` — `F` dominates (F&N=F); result is `N` only when there is no `F` and at
least one `N`. `OR` — `T` dominates (T|N=T); result is `N` only when there is no `T` and
at least one `N`. `not`: `¬T=F`, `¬F=T`, `¬N=N`.

### 4.3 Aggregations — `skipna` by default

Nulls are skipped; over the **non-null** subset:

| agg | all-valid | all-null / empty | notes |
|---|---|---|---|
| `sum` | sum of non-null | **0** (numeric identity) | a valid `NaN` in the data → result `NaN`. |
| `mean` | sum/count(non-null) | **null** (0/0) | |
| `min`/`max` | over non-null | **null** | IEEE compare; NaN not selected unless all values NaN. |
| `count` | # non-null | 0 | see §4.4 |
| `nunique` | # distinct non-null values | 0 | nulls are **not** counted as a distinct value. |
| `std`/`var` | sample, **ddof = 1**, over non-null | **null** if fewer than 2 non-null | |
| `first`/`last` | first/last **non-null** (by row order) | **null** | skipna; use out-valid flag (`wasm-abi.md` §9). |

### 4.4 `count` vs `size`

- **`count`** = number of **non-null** entries (per column). `count_null(vp,len)`.
- **`size`** = number of **rows**, nulls included; identical across all columns of a
  frame. In `groupby`, `size` = rows per group (incl. nulls); `count` = non-null per
  group per column. `df.shape[0]` uses `size`.

### 4.5 Null utilities & filtering

- **`fillNull(value)`** — replace null with `value`, set validity `1`; output all-valid.
  `value` must match the column dtype exactly (§3 has no implicit conversion for the
  fill value). Kernel: `fill_null_dt`.
- **`isNull()`** — boolean column, `true` where the source is null; the result itself has
  **no** nulls. `notNull` = `not(isNull)`. Kernel: `is_null`.
- **`filter(predicate)`** — keeps rows where the boolean/mask value is **`T` (valid &
  true)**. Rows where the predicate is `F` **or `N` (null)** are **dropped** (pandas
  boolean-indexing behavior: null predicate → excluded).
- **Grouping keys:** null key values form their own group (a single "null group"); they
  are not dropped from `groupby` (only `filter` drops them). Join on a null key does
  **not** match (null ≠ null for equijoin), consistent with SQL.

### 4.6 Ordering — `argsort` / `topk` (v1.1 addendum, orchestrator)

- **Total order per dtype (ascending):** numeric order (signed `i32`, unsigned `u32`,
  IEEE for floats) over valid values, with **`NaN` after `+inf`** (all NaN bit patterns
  compare equal to each other), and **null after everything** (nulls sort **last**).
- **Descending (`desc=1`)** reverses the *value* total order (so `NaN` comes first,
  being the largest value) — **nulls still sort last** in both directions
  (pandas `na_position='last'`).
- **Stability:** equal keys — including NaN ties and null ties — preserve original row
  order, in both directions.
- **Multi-key sort** = repeated stable single-key argsort from **last key to first**,
  threading the permutation through the `inout_perm` parameter (`wasm-abi.md` §9 C).
- **`topk`:** indices of the `k` extreme **valid** values under the same total order
  (nulls excluded; `NaN` participates as the largest value). `largest=1` → k largest,
  output ordered descending; `largest=0` → k smallest, ascending. Ties: lower original
  index first. Writes `min(k, non-null count)` indices.
- **`nunique`:** `NaN` counts as **one** distinct value; nulls are not counted (§4.3).

---

## 5. Quick reference for the expression compiler (Phase 3)

- Determine result dtype via §3.1; if the pair is unsupported-mixed, raise the §3 error.
- Emit the arithmetic/comparison **data** kernel + a `validity_and`/unary validity copy
  for propagation (`wasm-abi.md` §5.5, §8).
- Insert explicit `cast_*` kernels only where the user wrote `.cast()` or where §3.1
  mandates int→float widening; identity casts are elided.
- Aggregations lower to the `*_null` reduction kernels (§4.3); `count`/`size` are
  distinct (§4.4).
