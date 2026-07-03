# ADR-010 — Temporal dtypes: date32, timestamp, and timezone metadata

**Status:** accepted
**Date:** 2026-07-03
**Deciding agent:** v2 LEAD (bead dataframe-dh9.1), commissioned by Dylan for 0.2.0.
**Reverses:** the v1 non-goal "dates/timestamps/timezones (v2)" (spec §0 non-goals;
`dtypes.md` §1 "Non-goals (v2)").
**Depends on:** ADR-009 (i64) — `timestamp` is physically an `i64`.

---

## Context

v1 deferred all temporal types (spec §0). The parking lot (`status.md`) named two
blockers: the i64 encoding prerequisite (a timestamp needs 64-bit range) and a
timezone-database size concern. ADR-009 removes the first; this ADR removes the second
by keeping **all** timezone rules as thin metadata + the runtime's built-in `Intl`
database — we ship **no** tz database and **no** new wasm code.

We reverse the non-goal now because temporal columns are the single most common typed
column in real CSV/Arrow/Parquet data, and because the design can be almost entirely
**a registry mapping over the existing i32/i64 kernels** rather than a new kernel family.

## Decision

Add two logical temporal dtypes plus optional timezone metadata. Both follow the
**Apache Arrow** model: a logical type over a physical integer, with tz as column-level
metadata that never changes stored values.

### The two dtypes

| logical dtype | physical | unit / epoch | storage | validity |
|---|---|---|---|---|
| `date32` | `i32` | **days** since Unix epoch (1970-01-01), proleptic Gregorian | `Int32Array`, 4 B | Arrow LSB bitmap |
| `timestamp` | `i64` | **milliseconds** since Unix epoch, **always UTC** | `BigInt64Array`, 8 B | Arrow LSB bitmap |

- `date32` range ≈ ±5.88 million years; `timestamp` (i64 ms) range ≈ ±2.9×10^8 years.
- **`timestamp` values are always stored in UTC.** There is no "local" storage form.
- Null semantics are **unchanged**: null is the validity bit, never a sentinel day/ms.

### Timezone = column metadata (Arrow model)

A `timestamp` column *may* carry an optional **IANA timezone string** (e.g.
`"America/New_York"`, `"UTC"`, or a fixed offset like `"+05:30"`) as column metadata
(`Column.tz`, see `memory.d.ts` v2). This metadata:

- **Never changes the stored value** — the physical i64 is UTC ms regardless of `tz`.
- Affects **only** display (`toString`/`describe`) and the **tz-aware dt accessors**.
- Absent `tz` (`null`/`undefined`) → the column is displayed and its accessors compute
  in **UTC** (the default).

`date32` carries **no** tz (a calendar date is tz-independent by construction).

### No new wasm kernels — a logical→physical registry

Temporals add **zero** wasm exports. The dtype registry maps each logical dtype to a
**physical kernel token**, and all value-comparison / ordering / relational ops dispatch
to the existing kernel of that token:

| logical dtype | kernel token | reuses (per ABI §9) |
|---|---|---|
| `date32` | `i32` | `eq/ne/gt/ge/lt/le_i32_mask`, `argsort_i32`, `topk_i32`, `filter_i32`, `gather_i32`, `min/max/first/last_i32_null`, `hash_i32`, `group_build`, `join_hash_*` |
| `timestamp` | `i64` | the ADR-009 `*_i64` equivalents of the above |

Compare, sort (total order = physical integer order, **nulls last**, `dtypes.md` §4.6),
group, join, min/max/first/last, filter, and gather are therefore **already
implemented** — the frame/expr layer relabels the result dtype (e.g. `min` of a
`timestamp` column returns a `timestamp`, not a raw i64).

### Arithmetic is restricted (a small closed algebra)

Temporal arithmetic is **not** general integer arithmetic; only these forms are legal
(everything else is a **dtype error naming the op**, e.g.
`"cannot mul(timestamp, timestamp)"`):

| operation | result dtype | meaning / lowering |
|---|---|---|
| `timestamp − timestamp` | `i64` (duration, ms) | `sub_i64` on the physical values |
| `timestamp ± integer-ms` | `timestamp` | `add_i64`/`sub_i64`; the offset is i64 ms (i32/`number` widen to i64) |
| `date32 − date32` | `i32` (duration, days) | `sub_i32`; symmetric analog of the timestamp rule |
| `date32 ± integer-days` | `date32` | `add_i32`/`sub_i32`; offset is i32 days (`number` → i32) |

- Subtracting two absolutes yields a **plain integer duration** (i64 ms / i32 days),
  *not* another temporal — durations are not a distinct dtype in v2.
- `timestamp ± timestamp`, `date32 ± date32` (addition), `timestamp ± date32`,
  `mul/div/mod` on any temporal, and temporal ⊕ float all **error**. The offset in the
  `±integer` forms wraps in its physical width (i64/i32) like ordinary integer add — no
  trap. Null propagates (`out_valid = a_valid & b_valid`) exactly as for the underlying
  integer op.
- Scalar literals: `col('ts').add(1000)` adds **1000 ms**; `col('d').add(7)` adds
  **7 days**. A `number` literal follows the i64/i32 safe-int rule of ADR-009.

### `dt` accessors — JS-side, vectorized, no per-row `Date`

Field extraction (`.dt.year`, `.month`, `.day`, `.hour`, `.minute`, `.second`,
`.millisecond`, `.weekday`, `.dayOfYear`, `.quarter`) is computed **in JS over the
typed-array view**, with **no `Date` object per row**:

- **UTC path (default / no `tz`):** integer **civil-from-days** math (Howard Hinnant's
  algorithm) converts the day number to `(year, month, day)`; time-of-day fields come
  from the ms-of-day via integer `divmod` on `86_400_000`. For `timestamp` the day
  number is `floorDiv(ms, 86_400_000)` (negative-safe floor, see the cast worked
  example below) and ms-of-day is the corresponding non-negative remainder. No
  `Date`, no locale, branch-light — suitable for 1M+ rows.
- **tz-aware path (`tz` present):** a **cached `Intl.DateTimeFormat`** (one per tz
  string, memoized) formats each instant into the target zone's local fields. This is
  correct across DST/offset history (that is the whole point of using `Intl`) but is
  slower than the UTC integer path; it is documented as such. Only `timestamp` has a
  tz-aware path; `date32` accessors are always the plain civil-from-days fields.
- **Null in → null out**: a null row yields a null accessor result (validity copied).

**Weekday numbering — ISO 8601: Monday = 1 … Sunday = 7.** Stated explicitly and
locked. Derivation from the epoch day number `d` (1970-01-01 is a Thursday):

```
let w = (d + 3) % 7;   // JS % can be negative
if (w < 0) w += 7;     // negative-safe
weekday = w + 1;       // 1=Mon … 7=Sun
// checks: d=0 (Thu 1970-01-01)→4 · d=3 (Sun)→7 · d=4 (Mon)→1 · d=-4 (Sun 1969-12-28)→7
```

(ISO chosen over JS `Date.getUTCDay()`'s Sunday=0..6 for international-standard
unambiguity and alignment with `dayOfYear`/quarter; downstream code must not assume
pandas' Monday=0.)

### Casts

Temporal casts split into **free reinterprets** and **scale casts** (full cells and a
worked example in `dtypes.md` §2 v2):

- **`date32 ↔ i32`** and **`timestamp ↔ i64`**: **free reinterpret** (explicit,
  lossless, no value change). `date32 → i32` exposes the day count; `timestamp → i64`
  exposes epoch ms (tz metadata is **dropped** — i64 is a plain integer);
  `i32 → date32` / `i64 → timestamp` reinterpret the bits (result is UTC, no tz).
- **`date32 → timestamp` = × 86_400_000** (days → ms, midnight UTC of that date).
  Exact and overflow-free (`|days| < 2^31`, so `|ms| < 2^58 < 2^63`). Lowered to
  existing kernels: `cast_i32_i64` (sign-extend) then `mul_i64_scalar(86_400_000)`.
- **`timestamp → date32` = floorDiv(ms, 86_400_000)**. Uses **floor** division, not
  truncation, so instants before the epoch map to the correct (earlier) calendar day.
  Because `div_i64` truncates toward zero, this cast is done **JS-side** over the
  BigInt view (not a kernel). Negative worked example:
  - `ts = -1` ms (1969-12-31 23:59:59.999 UTC): `floorDiv(-1, 86400000) = -1` →
    `date32 = -1` = **1969-12-31**. (Truncation would give `0` = 1970-01-01, wrong.)
  - `ts = -86_400_001`: `floorDiv = -2` = **1969-12-30**.
  - `ts = 86_400_000`: `floorDiv = 1` = **1970-01-02**.

  JS reference: `let d = ts / 86_400_000n; if (ts % 86_400_000n !== 0n && ts < 0n) d -= 1n;`

Casting a temporal to/from any other numeric dtype (e.g. `timestamp → f64`) goes
through its physical type explicitly (`timestamp → i64 → f64`); there is no direct
`timestamp → f64` kernel-cast.

## Consequences

- `DType` gains `'date32'` and `'timestamp'`; `Column` gains an optional `tz` field
  (`memory.d.ts` v2). The **registry** already carries a per-dtype kernel token
  (`DTypeInfo.wasm`); for temporals it diverges from the dtype name for the first time
  (`DTYPES.date32.wasm === 'i32'`, `DTYPES.timestamp.wasm === 'i64'`) — this **is** the
  logical→physical mapping, so the frame/expr layer reads it directly.
- **No wasm binary grows** for temporals (zero new exports) — the size budget is
  untouched. i64 kernels (ADR-009) are the only new wasm; they must land first.
- The tz-database concern is fully sidestepped: the only tz source is the platform
  `Intl` DB (already present in every Node ≥ 18 and evergreen browser). We ship no
  zoneinfo. Cost of a tz-aware accessor = one cached `Intl.DateTimeFormat` per tz.
- Arrow/CSV/Parquet integration (dh9.6/dh9.7) maps cleanly: Arrow `Date[DAY]`↔`date32`,
  Arrow `Timestamp[MILLI, tz]`↔`timestamp`+tz; Parquet `DATE`/`TIMESTAMP(MILLIS)`
  likewise (ADR-011).
- Conformance fixtures (dh9.2) must cover: negative `timestamp → date32` floor cases,
  the ISO weekday mapping across a week (incl. pre-epoch), tz-aware vs UTC accessor
  divergence across a DST boundary, and every arithmetic-restriction error.

## Alternatives rejected

- **Store local time + a tz (drop the always-UTC rule):** breaks comparison/sort/join
  semantics (two equal instants in different zones would not compare equal) and forces
  tz math into the kernels. Arrow's always-UTC + tz-metadata model is the right one.
  Rejected.
- **Dedicated temporal wasm kernels (compare/sort/arith):** pure duplication of the
  i32/i64 kernels with a different name; grows every binary for no behavior change. The
  registry-token approach reuses them at zero wasm cost. Rejected.
- **`Date`-object-per-row accessors:** allocates millions of `Date`s and drags in
  locale/DST cost per value; fails the perf targets. Integer civil-from-days (UTC) +
  cached `Intl.DateTimeFormat` (tz) is the fast, correct split. Rejected.
- **Bundle a zoneinfo database:** blows the size budget and duplicates what `Intl`
  already provides. Rejected — use the platform DB.
- **`date64` / nanosecond timestamps:** `date64` (ms since epoch as a date) is
  redundant with `timestamp`; nanoseconds need i64 that overflows at ~year 2262 and add
  little for analytics. Millisecond `timestamp` + `date32` covers the v2 need; finer
  units are a future ADR. Rejected for v2.
- **Making duration a distinct dtype:** a `timestamp − timestamp` result is just an
  integer count of ms; introducing a `duration` dtype adds a type with almost no unique
  behavior. Return `i64`/`i32` and let the user label it. Rejected for v2.
