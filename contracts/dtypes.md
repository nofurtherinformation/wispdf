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

---

# v2 additions — i64, date32, timestamp (orchestrator-owned)

**Status:** v2, authoritative. Companion to ADR-009 (i64), ADR-010 (temporals). Extends
the v1 sections above; v1 rules that are not restated here are **unchanged**. Adds three
dtypes (`i64`, `date32`, `timestamp`), the full new cast pairs, the broadened
implicit-widening lattice, the temporal-arithmetic restriction table, and `dt` accessor
semantics. Read-only to subagents.

## 6. Dtypes v2

| dtype | storage | bytes/value | scalar wasm type | JS boundary | kernel token | validity | notes |
|---|---|---|---|---|---|---|---|
| `i64` | contiguous i64 | 8 | `i64` | **`bigint`** (`BigInt64Array` view) | `i64` | bitmap | signed 64-bit. Wrapping arithmetic (mod 2^64). ADR-009. |
| `date32` | contiguous i32 | 4 | `i32` | `number` (days) | **`i32`** | bitmap | **logical**: days since 1970-01-01 UTC, proleptic Gregorian. Physical = i32. ADR-010. |
| `timestamp` | contiguous i64 | 8 | `i64` | **`bigint`** (ms) | **`i64`** | bitmap | **logical**: ms since 1970-01-01, **always UTC**; optional IANA `tz` **metadata** (display/accessors only). Physical = i64. ADR-010. |

- **Kernel token = physical dtype** (the logical→physical registry, ADR-010): `date32`
  dispatches to `i32` kernels, `timestamp` to `i64` kernels. This is `DTypeInfo.wasm`
  in `memory.d.ts`; for these dtypes it **differs** from the dtype name (the first time
  it does).
- **Null representation is unchanged** (validity bitmap; never a sentinel day/ms/value).
- `i64` introduces the **only** new `TypedArray`, `BigInt64Array`. Its `NaN`/`float`
  concerns do **not** apply (integer) — `DTypeInfo.float === false` for `i64`, `date32`,
  and `timestamp`.

## 7. Cast matrix v2 — the new pairs

Casting stays **explicit only**. Legend as §2 (**=** lossless · **≈** may lose precision
· **⚠** out-of-range/invalid → **null** · **id** identity · **✗** throws). These pairs
**extend** the §2 matrix; the numeric core now includes `i64` as a row and a column.

### 7.1 To / from `i64` (numeric core extension)

**Casts TO `i64`** (new column of the §2 matrix):

| from → `i64` | semantics |
|---|---|
| `f64` → `i64` | ⚠ truncate toward zero; `\|x\| ≥ 2⁶³` or `NaN` → **null** |
| `f32` → `i64` | ⚠ (as `f64→i64`) |
| `i32` → `i64` | = exact (**sign-extend**) |
| `u32` → `i64` | = exact (**zero-extend**) |
| `bool` → `i64` | = (`t→1n`, `f→0n`) |
| `i64` → `i64` | id |
| `utf8` → `i64` | ✗ (throws) |

**Casts FROM `i64`** (new row of the §2 matrix):

| `i64` → to | semantics |
|---|---|
| `i64` → `f64` | ≈ exact if `\|x\| ≤ 2⁵³`, **else ROUNDS to nearest f64 (NOT null)** |
| `i64` → `f32` | ≈ (as `i64→f64→f32`; rounds) |
| `i64` → `i32` | **wrap-truncate** low 32 bits, signed (never null/trap) |
| `i64` → `u32` | **wrap-truncate** low 32 bits, unsigned (never null/trap) |
| `i64` → `bool` | `x≠0 → true`, `0 → false` |
| `i64` → `utf8` | ✗ (throws) |

Load-bearing rules for the `i64` casts:

- **`f64→i64` / `f32→i64` (⚠):** truncate toward zero; if `|x| ≥ 2⁶³` (out of i64
  range) or `x` is `NaN`, the output validity bit is `0` (**null**). Never traps.
  Same failure model as `f64→i32`.
- **`i64→f64` / `i64→f32` (≈, NEVER null):** exact when `|x| ≤ 2⁵³`; for larger
  magnitudes the value **rounds to the nearest representable float** and the row stays
  **valid** (matches `Number(bigint)`). Widening casts lose precision but do **not**
  introduce nulls — do not add a range-null here.
- **`i64→i32` / `i64→u32` (wrap-truncate, NEVER null):** keep the low 32 bits, reinterpret
  as signed / unsigned (two's complement). E.g. `0x1_0000_0000n → 0`; `-1n → -1` (i32) /
  `4294967295` (u32). No trap, no null. **Note:** this differs from v1's `i32→u32`
  range-null cast — i64↔narrow-int deliberately follows i64's wrapping-arithmetic model
  (ADR-009), because 64→32-bit narrowing has a well-defined wrap and no useful "range
  failure" notion.
- **`i32→i64` / `u32→i64` (=):** exact widening (sign-extend / zero-extend).
- **`bool→i64` / `i64→bool`:** mirror the v1 bool↔i32 rows.
- **Null propagates** through every i64 cast (validity bit copied), in addition to any
  ⚠ range-nulls.

### 7.2 Temporal casts — reinterpret + scale

Temporal casts are **explicit**. Two kinds:

| cast | kind | semantics |
|---|---|---|
| `date32 ↔ i32` | **free reinterpret** | lossless; `date32→i32` = day count, `i32→date32` = same bits as a date. No value change. |
| `timestamp ↔ i64` | **free reinterpret** | lossless; `timestamp→i64` = epoch ms (**tz metadata dropped**), `i64→timestamp` = same bits (UTC, no tz). |
| `date32 → timestamp` | **scale ×** | value `= days × 86_400_000` (ms, midnight UTC). Exact (\|days\|<2³¹ ⇒ \|ms\|<2⁵⁸). Lowered to `cast_i32_i64` then `mul_i64_scalar(86_400_000)`. |
| `timestamp → date32` | **scale ÷ (floor)** | value `= floorDiv(ms, 86_400_000)`. **Floor**, not truncation. JS-side over the BigInt view (no floor-div kernel; `div_i64` truncates). |

Any temporal ↔ other-numeric cast (e.g. `timestamp → f64`) must route through the
physical type explicitly (`timestamp → i64 → f64`); there is no direct kernel-cast.
Null propagates.

**Negative floor-div worked example (`timestamp → date32`)** — the reason floor ≠ trunc:

| `ts` (ms) | wall clock (UTC) | `floorDiv(ts, 86_400_000)` | `date32` → date | (trunc would give) |
|---|---|---:|---|---:|
| `-1n` | 1969-12-31 23:59:59.999 | `-1` | **1969-12-31** | `0` = 1970-01-01 ✗ |
| `-86_400_001n` | 1969-12-30 23:59:59.999 | `-2` | **1969-12-30** | `-1` ✗ |
| `-86_400_000n` | 1969-12-31 00:00:00.000 | `-1` | **1969-12-31** | `-1` ✓ (exact multiple) |
| `86_400_000n` | 1970-01-02 00:00:00.000 | `1` | **1970-01-02** | `1` ✓ |
| `86_400_500n` | 1970-01-02 00:00:00.500 | `1` | **1970-01-02** | `1` ✓ |

JS reference: `let d = ts / 86_400_000n; if (ts % 86_400_000n !== 0n && ts < 0n) d -= 1n;`
(BigInt `/` truncates toward zero; the correction subtracts 1 for a negative
non-exact instant).

## 8. Implicit-conversion lattice v2 (mixed arithmetic)

v1's single rule (int→float widening) broadens to a **widening lattice**. `add sub mul`
and `div`/`mod` (§3.2 rules unchanged: int⊕int stays int, truncating; zero divisor →
null) use it. The complete allowed mixed pairs (append to §3.1):

| left ⊕ right | result | conversion |
|---|---|---|
| `i64` ⊕ `i64` | `i64` | none |
| `i32` or `u32` ⊕ `i64` | `i64` | int widened to `i64` (**exact**; i32 sign-extend, u32 zero-extend) |
| `i64` ⊕ `f64` | `f64` | i64 widened to f64 (⚠ rounds if \|x\|>2⁵³) |
| `i64` ⊕ `f32` | **`f64`** | **both** widened to f64 (i64→f64 rounds >2⁵³; f32→f64 exact). Result is **f64, not f32** — an f32 mantissa cannot hold i64. |

Still an **error** (unchanged): `i32`⊕`u32`, `f32`⊕`f64`, `i64`⊕`bool`, `bool`⊕anything,
`utf8`⊕anything, and any **temporal** combined with a numeric except the temporal forms
in §9. `neg` preserves dtype (including `i64`, wrapping).

## 9. Temporal arithmetic — restricted algebra

Only these forms are legal; **everything else raises a dtype error naming the op**
(e.g. `"unsupported op mul(timestamp, timestamp)"`). Offsets wrap in their physical
width (no trap); null propagates (`out_valid = a_valid & b_valid`).

| operation | result | lowering / meaning |
|---|---|---|
| `timestamp − timestamp` | `i64` (duration ms) | `sub_i64` |
| `timestamp + int-ms` / `timestamp − int-ms` | `timestamp` | `add_i64`/`sub_i64`; offset is i64 ms (i32/`number` widen to i64) |
| `date32 − date32` | `i32` (duration days) | `sub_i32` |
| `date32 + int-days` / `date32 − int-days` | `date32` | `add_i32`/`sub_i32`; offset is i32 days (`number` → i32) |

Errors (non-exhaustive, all "everything else"): `timestamp + timestamp`,
`date32 + date32`, `timestamp ± date32`, `mul/div/mod` on any temporal, `timestamp ⊕
float`, `date32 ⊕ float`. Scalar literals adopt the unit: `.add(1000)` on a `timestamp`
= +1000 ms; on a `date32` = +1000 days (`number` literal follows the ADR-009 safe-int
rule; `bigint` allowed for ms).

## 10. `dt` accessors — semantics

Field extraction on `date32`/`timestamp`, computed **JS-side, vectorized, no per-row
`Date`** (ADR-010). Fields: `year month day hour minute second millisecond weekday
dayOfYear quarter`. (For `date32`, time-of-day fields are `0`.)

- **UTC is the default** (no `tz` metadata → interpret the instant in UTC via integer
  civil-from-days math; time-of-day via integer `divmod` on `86_400_000`, with a
  **negative-safe floor** to derive the day number of a `timestamp`, per §7.2).
- **tz metadata behavior:** a `timestamp` column may carry an optional IANA `tz` string.
  When present, accessors return **local** fields for that zone via a **cached
  `Intl.DateTimeFormat`** (one per tz, memoized; DST/offset-correct, slower than the
  UTC integer path). The stored value never changes; only the *view* does. `date32` has
  no tz.
- **Weekday numbering = ISO 8601: Monday = 1 … Sunday = 7** (locked; not JS 0–6, not
  pandas 0–6). Derivation: `w = (dayNum + 3) mod 7` (negative-safe → `+7`), `weekday =
  w + 1` (1970-01-01 is Thursday = 4).
- `quarter` ∈ `1..4`; `dayOfYear` ∈ `1..366`; `month` ∈ `1..12`; `day` ∈ `1..31`;
  `hour` ∈ `0..23`; `minute`/`second` ∈ `0..59`; `millisecond` ∈ `0..999`.

## 11. Null / NaN interactions for every new op

- **i64 has no `NaN`** (integer): the §4 "null ≠ NaN" caveat is moot for `i64`,
  `date32`, `timestamp`. Null is *only* the validity bit.
- **Arithmetic / comparison (i64 & temporals):** binary op `out_valid = a_valid &
  b_valid`; unary copies validity (§4.1 unchanged). i64 comparison masks follow the
  same mask+validity convention (§4.1); a null operand → mask bit `0`, row null, dropped
  by `filter` (§4.5).
- **`div_i64`/`mod_i64` zero divisor → null** (validity `0`), never trap (§3.2 model).
- **Casts:** null propagates through every new cast; ⚠ casts (`f64/f32→i64`) add
  range-nulls; widening/wrap casts (`i64→f64/f32/i32/u32`) add **no** nulls.
- **Reductions (i64 & temporals, `skipna`):** over non-null values —
  - `sum_i64` → wrapping i64 sum; all-null/empty → `0n` (additive identity).
  - `mean_i64` → f64 (precision-lossy >2⁵³); all-null/empty → **null** (`NaN` at kernel,
    mapped via `count_null`).
  - `min_i64`/`max_i64` (and `timestamp`/`date32` min/max via i64/i32) → over non-null;
    all-null/empty → **null** (kernel returns `0`/`0n`; caller consults `count_null`,
    ABI §9 v2).
  - `std_i64`/`var_i64` → f64, ddof=1; fewer than 2 non-null → **null**.
  - `first_i64`/`last_i64` → first/last **non-null** (skipna); all-null → out-valid `0`
    → **null**.
  - `count` = non-null (dtype-agnostic `count_null`); `size` = rows incl. nulls (§4.4
    unchanged). `nunique_i64` counts distinct non-null values; nulls not counted.
- **`dt` accessors:** null row → null result (validity copied). A tz-aware accessor on a
  valid instant is always valid (no null introduced).
- **Grouping / join:** null temporal/i64 keys form one null group and never join-match,
  identical to §4.5 (null keys hash to `H_NULL`, ABI §9 Agent D).

## 12. Expression-compiler quick reference (v2 delta)

- Result dtype via §8 (extended lattice); temporal ops via §9 (restricted algebra) —
  emit the dtype error for any unlisted temporal/i64 mixed pair.
- Temporal value-ops dispatch to the **physical kernel token** (`date32`→`i32`,
  `timestamp`→`i64`); relabel the result dtype back to the logical type (e.g. `min` of a
  `timestamp` returns a `timestamp`; `timestamp − timestamp` returns `i64`).
- Scale-casts: `date32→timestamp` lowers to `cast_i32_i64` + `mul_i64_scalar`;
  `timestamp→date32` is JS-side floor-div (§7.2). Reinterpret casts emit no kernel.
- `dt` accessors are JS-side (no kernel); tz-aware ones use the cached
  `Intl.DateTimeFormat`.

## §13 — `str` namespace v1 (census-perf, 2026-07-03)

- `col('g').str.slice(start, end?)` → `utf8`. **JS `String.prototype.slice` semantics**
  (negative indices from the end; `end` omitted = to end; out-of-range clamps; UTF-16
  code-unit indexing, same as every JS string API — document the surrogate-pair caveat).
- Null propagation: null rows stay null; the operation never inspects null rows.
- **Implementation contract:** applied to the **dictionary values once** (O(unique)),
  producing sliced values that are re-deduplicated into the result dictionary; row
  indices are remapped, never iterated per-row on the JS side beyond the index remap.
- Other `str` ops (len, upper/lower, startsWith, …) stay in the v2 parking lot until a
  workload needs them (YAGNI); they must follow the same dictionary-values contract.
