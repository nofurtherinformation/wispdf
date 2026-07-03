# databonk

Columnar dataframe library for JavaScript — pandas-familiar API, WebAssembly-accelerated kernels, zero-copy typed-array views, dual ESM/CJS, TypeScript types.

```
npm install databonk
```

Works in **Node.js ≥ 18**, **Vite**, and **webpack** without configuration gymnastics.
Each `.wasm` binary is ≤ 23 KB gzipped; the JS entry is **27.0 KB gzipped** (0.2.0,
v2 surface including i64 + temporals; see [Bundle sizes](#bundle-sizes) and [ADR-012](docs/adr/ADR-012-entry-size-budget-v2.md)).

---

## Why databonk?

Most JS "dataframe" libraries either wrap pandas in WASM (giant binary) or operate
row-by-row over plain objects (slow).  databonk takes a different path:

- **Columns live in WASM linear memory.** JS holds zero-copy `TypedArray` views over
  those buffers — no marshalling overhead on the hot path.
- **SIMD kernels** (auto-detected; scalar fallback for older Safari). Core
  operations are 3–4× faster than Arquero on 1M-row pipelines.
- **pandas-shaped API** (`filter`, `groupby().agg()`, `sortValues`, `join`) without
  pandas' index-alignment surprises.
- **Opt-in threads** via `enableThreads()` (COOP/COEP required in browser; always
  available in Node). 3.3–3.5× speedup on 4 workers for 10M-row reductions.
- **I/O** — CSV, JSON records, Arrow IPC, and now Parquet (`databonk/parquet` subpath).
- **v2 dtypes** — `i64`/BigInt columns, `date32`, `timestamp` with timezone metadata,
  and `dt` accessor proxy (`.year()`, `.month()`, `.weekday()`, …).

### v1 non-goals (honest)

The following remain deferred or out of scope:

- `pandas.Index` / automatic alignment (row position is identity)
- Chunked / out-of-core columns; lazy query optimizer
- Write-side CSV formatting; mutation-in-place API (all ops return new frames)
- wasm64 (> 4 GB) — under consideration

i64, dates/timestamps/timezones, and Parquet I/O **shipped in v0.2.0**.

---

## Install

```
npm install databonk
```

Peer-required: **Node.js ≥ 18** (or a modern browser with WebAssembly support).

---

## Quickstart

```typescript
import { init, DataFrame, col } from 'databonk';

// Load the wasm runtime once at startup (auto-detects SIMD)
await init();

// Build a frame from typed arrays or JS arrays
const df = DataFrame.fromColumns({
  id:    new Int32Array([1, 2, 3, 4, 5]),
  value: new Float64Array([10.5, 3.2, 8.1, 5.9, 2.7]),
  group: ['a', 'b', 'a', 'b', 'a'],
});

// Filter (expression path — fast, WASM-compiled)
const filtered = df.filter(col('value').gt(5));

// Group by + aggregate
const summary = filtered.groupby(['group']).agg({ value: ['sum', 'mean'] });

// Export to JS objects
const records = summary.toRecords();
console.log(records);
// [ { group: 'a', value_sum: 18.6, value_mean: 9.3 },
//   { group: 'b', value_sum:  5.9, value_mean: 5.9 } ]

// Clean up WASM memory when done
filtered.dispose();
summary.dispose();
df.dispose();
```

### scope() — automatic cleanup

```typescript
import { scope } from 'databonk';

const result = scope(() => {
  const filtered = df.filter(col('value').gt(5));
  const grouped  = filtered.groupby(['group']).agg({ value: 'sum' });
  return grouped.toRecords();   // primitive — safe to return outside scope
});
// all intermediate frames disposed automatically
```

### I/O

```typescript
import { init, DataFrame, fromCSV, fromArrow, toArrow, fromJSON, toJSON } from 'databonk';

// Load the wasm runtime once at startup (needed for fromArrow)
const rt = await init();

// CSV (auto-infers dtypes)
const df = fromCSV(csvText, { delimiter: ',', header: true });

// Arrow IPC (compatible with Apache Arrow; no runtime arrow dep needed)
const buf = toArrow(df);
const df2 = fromArrow(buf, rt);

// JSON records
const df3 = DataFrame.fromRecords([{ x: 1, y: 'a' }, { x: 2, y: 'b' }]);
const json = toJSON(df3);
```

---

## i64 / BigInt columns

> **ADR-009** — reverses the v1 non-goal "i64 / BigInt columns".

`i64` columns store signed 64-bit integers in WASM linear memory as a contiguous
`BigInt64Array` — the same zero-copy `viewOf` model as every other dtype. BigInt
cost appears only at scalar crossings (literals, reduction returns, row-proxy access).

### Construction

```typescript
import { init, DataFrame, col, lit } from 'databonk';

await init();

// BigInt64Array fast path — zero-copy into WASM memory
const df = DataFrame.fromColumns({
  id:    BigInt64Array.from([1n, 2n, 9007199254740993n]),
  score: new Float64Array([0.1, 0.9, 0.5]),
});
// dtypes: { id: 'i64', score: 'f64' }

// Plain arrays auto-detect i64 when bigint values are present
const df2 = DataFrame.fromColumns({
  big: [1n, 2n, null, -3n],
}, { dtypes: { big: 'i64' } });

// Explicit dtype forces i64 from safe-integer numbers
const df3 = DataFrame.fromColumns(
  { count: [1, 2, 3] },
  { dtypes: { count: 'i64' } },
);
```

**Safe-int throw**: passing a `number` outside `Number.isSafeInteger` range
(`|x| > 2^53 − 1`) to an i64 column throws a descriptive `RangeError`.
Use a `bigint` literal for values near or above 2^53.

### Arithmetic, reductions, and precision caveat

```typescript
import { init, DataFrame, col, lit } from 'databonk';

await init();

const df = DataFrame.fromColumns({
  a: BigInt64Array.from([10n, 20n, 30n]),
  b: BigInt64Array.from([ 1n,  2n,  3n]),
});

// Arithmetic wraps on overflow (two's complement mod 2^64)
const added = df.withColumn('c', col('a').add(col('b')));

// Literal bigint in expressions
const scaled = df.withColumn('d', col('a').mul(lit(1_000_000n)));

// Groupby + agg: sum/min/max return bigint; mean returns number (f64)
const byGroup = DataFrame.fromColumns({
  g: ['x', 'y', 'x'],
  v: BigInt64Array.from([10n, 20n, 30n]),
});
const grouped = byGroup.groupby('g').agg({ v: ['sum', 'mean', 'min'] });
// v_sum: bigint, v_mean: number, v_min: bigint

added.dispose();
scaled.dispose();
grouped.dispose();
byGroup.dispose();
df.dispose();
```

> **Precision caveat (ADR-009):** `mean`, `std`, and `var` on an `i64` column convert
> each value to `f64` first. For magnitudes above 2^53 the conversion rounds to the
> nearest representable float; the result loses precision but never throws. This is the
> same trade-off v1 already makes for `mean_i32`. Use `sum` (returns `bigint`) when
> exact 64-bit integer results are required.

### Widening and casting

- `i32`/`u32` ⊕ `i64` → `i64` (exact widen)
- `i64` ⊕ `f64` → `f64` (rounds if |x| > 2^53)
- `i64` ⊕ `f32` → `f64` (both widen; f32 cannot hold an i64)
- `f64 → i64`: truncates toward zero; out-of-range or NaN → **null** (never traps)
- `i64 → i32`/`u32`: wraps to the low 32 bits (two's complement)

---

## Temporal dtypes

> **ADR-010** — reverses the v1 non-goal "dates, timestamps, timezones".

Two logical temporal dtypes, both following the Apache Arrow model:

| logical dtype | physical | unit / epoch |
|---|---|---|
| `date32` | `i32` | days since Unix epoch (1970-01-01), proleptic Gregorian |
| `timestamp` | `i64` | milliseconds since Unix epoch, **always UTC** |

`timestamp` values are stored in UTC; timezone is column-level metadata only and never
changes the stored value (Arrow model). Null semantics are unchanged: null is the
validity bit, never a sentinel day/ms.

### Construction

```typescript
import { init, DataFrame, col } from 'databonk';

await init();

// date32: days since 1970-01-01 stored as i32.
// Convert JS Date to day count: Math.floor(date.getTime() / 86_400_000)
// 19737 = 2024-01-15, 19783 = 2024-03-01
const dates = DataFrame.fromColumns(
  { d: new Int32Array([19737, 19783, 0]) },
  { dtypes: { d: 'date32' } },
);

// timestamp: BigInt64Array of UTC milliseconds since epoch
// 1717243200000n = 2024-06-01T12:00:00Z  (new Date(...).getTime())
const ts = DataFrame.fromColumns(
  { ts: BigInt64Array.from([1717243200000n, 1718409600000n]) },
  { dtypes: { ts: 'timestamp' } },
);

// Attach IANA timezone metadata (display / dt accessors only; stored value stays UTC)
const tsWithTz = DataFrame.fromColumns(
  { ts: BigInt64Array.from([1717243200000n, 1718409600000n]) },
  { dtypes: { ts: 'timestamp' }, tzs: { ts: 'America/Chicago' } },
);

dates.dispose();
ts.dispose();
tsWithTz.dispose();
```

### `dt` accessor proxy

Field extraction runs entirely in JS over the typed-array view — no `Date` object per row.
UTC path uses integer civil-from-days math; tz-aware path uses a cached `Intl.DateTimeFormat`.

```typescript
import { init, DataFrame, col } from 'databonk';

await init();

const tsData = DataFrame.fromColumns(
  { created: BigInt64Array.from([1717243200000n]) },
  { dtypes: { created: 'timestamp' }, tzs: { created: 'America/Chicago' } },
);

// Expression path: col('created').dt.year() → Expr producing i32 Series
const years  = tsData.withColumn('yr',  col('created').dt.year());
const months = tsData.withColumn('mo',  col('created').dt.month());
const wkdays = tsData.withColumn('wd',  col('created').dt.weekday()); // ISO: Mon=1…Sun=7

// Series path (Series.dt returns SeriesDtProxy)
const series = tsData.col('created');
if (series) {
  const yearSeries = series.dt.year();  // new Series, dtype=i32
  // yearSeries is a view; no separate dispose() needed
  void yearSeries;
}

years.dispose();
months.dispose();
wkdays.dispose();
tsData.dispose();
```

Available accessors (all return i32 values): `year`, `month`, `day`, `hour`, `minute`,
`second`, `millisecond`, `weekday` (ISO 8601: Mon=1…Sun=7), `dayOfYear`, `quarter`.

`hour`, `minute`, `second`, and `millisecond` are not available on `date32` columns
(calendar date has no time-of-day component) and throw a `TypeError` if attempted.

tz-aware accessors (`ts.dt.*` with a tz-tagged `timestamp` column) use the platform
`Intl` timezone database — correct across DST and offset history. The UTC path (no tz
or `tz='UTC'`) is faster (integer math, branch-light, suitable for 1M+ rows).

### Restricted temporal arithmetic

Only these forms are legal; everything else is a typed `FrameError`:

| operation | result dtype | notes |
|---|---|---|
| `timestamp − timestamp` | `i64` (ms) | duration; reuses `sub_i64` |
| `timestamp + integer` | `timestamp` | offset is i64 ms; wraps mod 2^64 |
| `timestamp − integer` | `timestamp` | same |
| `date32 − date32` | `i32` (days) | reuses `sub_i32` |
| `date32 + integer` | `date32` | offset is i32 days |
| `date32 − integer` | `date32` | same |

`mul`, `div`, `mod` on any temporal; `timestamp + date32`; `timestamp + timestamp` (addition);
and any temporal ⊕ float all throw a descriptive error.

---

## Parquet I/O

> **ADR-011** — reverses the v1 non-goal "Parquet I/O (Arrow IPC only)".

Parquet support ships as a **separate subpath** so the main entry stays dep-free
and under 30 KB. Only importing `databonk/parquet` pulls in the runtime dependencies.

```typescript
import { init, defaultRuntime } from 'databonk';
import { readParquet, writeParquet } from 'databonk/parquet';

await init();
const rt = defaultRuntime();

// Read a Parquet file (async — hyparquet reader is Promise-based)
const bytes: Uint8Array = new Uint8Array(0); // replace with actual file bytes
const dfP = await readParquet(bytes, rt);

// Write a Parquet file (sync; default uncompressed, or 'snappy')
const out: Uint8Array = writeParquet(dfP, { compression: 'snappy' });

dfP.dispose();
```

### Supported dtype profile (ADR-011)

| databonk dtype | Parquet physical / logical |
|---|---|
| `f64` | `DOUBLE` |
| `f32` | `FLOAT` |
| `i32` | `INT32` |
| `u32` | `INT32` + `UINT_32` converted type |
| `i64` | `INT64` |
| `bool` | `BOOLEAN` |
| `utf8` | `BYTE_ARRAY` + `STRING(UTF8)`, dictionary-encoded |
| `date32` | `INT32` + `DATE` logical |
| `timestamp` | `INT64` + `TIMESTAMP(MILLIS, isAdjustedToUTC)`; tz round-trips in `key_value_metadata` |

Compression: **Snappy** and **uncompressed** (both native to `hyparquet`).

Anything outside the profile raises a clear, specific `Error` naming what was
unsupported — never a silent wrong result. Explicitly out of profile: gzip/brotli/zstd/lz4
compression; `INT96` timestamps; nested/repeated (`LIST`/`MAP`/`STRUCT`) columns;
`FIXED_LEN_BYTE_ARRAY`; `DECIMAL`.

### Subpath packaging

Runtime dependencies (`hyparquet 1.26.2` + `hyparquet-writer 0.16.1`) are declared in
`package.json` `"dependencies"` and imported as external modules at runtime — they are
**not bundled into the tarball**. They install alongside databonk automatically. The main
entry (`.`) has zero runtime dependencies and is not affected.

---

## Expression API vs lambda escape hatch

databonk has two filter/map styles:

### Expression path (fast — WASM-compiled)

```typescript
// col() builds an AST; the compiler fuses compare→filter into one kernel call.
df.filter(col('value').gt(5).and(col('group').eq('a')));
df.withColumn('doubled', col('value').mul(2));
df.filter(col('value').isNull().not());
```

Expressions support: `add sub mul div mod neg`, `gt ge lt le eq ne`,
`and or not`, `isNull fillNull`, `cast`, and aggregations
`sum mean min max count nunique std var first last`.

### Lambda escape hatch (SLOW PATH — scalar JS speed)

> **Warning:** `filterFn` and `mapFn` iterate rows via a JS row-proxy. They avoid
> data copies but run at scalar JS speed, not WASM speed. Use expressions above
> whenever possible.

```typescript
// SLOW PATH — use only when an expression cannot express the logic
df.filterFn(r => r.value > 5 && r.group === 'a');

// Expression equivalent (fast):
df.filter(col('value').gt(5).and(col('group').eq('a')));
```

ADR-003 requires showing the expression equivalent alongside every lambda example —
this is not just documentation style; it's a reminder that the fast path exists.

---

## Benchmark table

Measured in **Docker (Debian bookworm), Node v22.23.1, Linux** (single thread, SIMD build).
Numbers are medians of 3 independent fresh runs at 1M rows.
Source: [`bench/baselines/e2e-v1.json`](bench/baselines/e2e-v1.json).

| Operation | databonk (ms) | Arquero (ms) | Ratio |
|---|---:|---:|---:|
| filter → groupby → sum (pipeline) | 12.3 | 45.3 | **3.7× faster** |
| join (inner, string key) | 68.6 | 120.6 | **1.8× faster** |
| sortValues (f64, 1M rows) | 145.3 | 248.9 | **1.7× faster** |

> **Caveats:** Arquero times include `.objects()` materialisation (that's what its
> pipeline naturally produces); databonk times do not materialise to JS objects.
> Results vary by machine, Node version, dataset shape, and JIT warm-up.
> Run `node bench/e2e/pipeline.mjs` after `npm run build` to reproduce.

Danfo.js is tracked in `/bench/baselines/danfo.json` for informational comparison
but is not shown here (it uses TensorFlow.js under the hood, making size
comparisons misleading).

### i64 kernel performance (v2.3)

Measured in Docker, Node v22.23.1, 1M elements, SIMD build.
Source: recorded run in [`bench/baselines/i64-threads-v1.txt`](bench/baselines/i64-threads-v1.txt) (`bench/kernels/i64.mjs`).

| Operation | WASM (ops/s) | JS BigInt64Array (ops/s) | Ratio |
|---|---:|---:|---:|
| `add_i64` @1M (gate ≥ 1.5×) | 4 447 | 1 555 | **2.9× faster** |
| `mul_i64` @1M | 4 175 | 1 534 | **2.7× faster** |
| `hash_i64` @1M | 1 724 | 50 | **34.6× faster** |

### Parallel mode (4 workers, 10M f64 elements)

Source: recorded run in [`bench/baselines/i64-threads-v1.txt`](bench/baselines/i64-threads-v1.txt) (`bench/workers/threads-bench.mjs`).

| Op | 1 thread (ms) | 4 workers (ms) | Speedup |
|---|---:|---:|---:|
| sum | 2.72 | 0.81 | 3.3× |
| mean | 2.75 | 0.80 | 3.4× |
| min | 4.48 | 1.24 | 3.6× |

See [docs/threads.md](docs/threads.md) for setup.

---

## Bundle sizes

Gzipped, 0.2.0 build (Docker, Debian bookworm, Node v22.23.1, 2026-07-03):

| Asset | Gzipped | Notes |
|---|---:|---|
| `dist/index.js` (ESM) | 27.0 KB | v2 surface (i64 + temporals); ADR-012 budget 30 KB |
| `dist/index.cjs` (CJS) | 27.3 KB | |
| `dist/simd.wasm` | 22.1 KB | |
| `dist/scalar.wasm` (fallback) | 17.4 KB | |
| `dist/simd-threads.wasm` (threads opt-in) | 21.9 KB | |
| `dist/parquet.js` (`databonk/parquet` ESM) | 20.9 KB | hyparquet + hyparquet-writer as external imports; not bundled |

The ADR-012 budget for the main entry is **30 KB gz**. The 27.0 KB reading is with the full
v2 surface imported. Consumers who tree-shake to a v1-profile import (no i64/temporal/parquet)
pay approximately the v1 entry cost via their bundler's dead-code elimination.

---

## Feature matrix

| Feature | v0.1.0 | v0.2.0 |
|---|---|---|
| f64 / f32 / i32 / u32 / bool / utf8 columns | yes | yes |
| i64 / BigInt columns | no | **yes (ADR-009)** |
| Date / timestamp / timezone | no | **yes (ADR-010)** |
| `dt` accessor proxy (year/month/…/weekday) | no | **yes (ADR-010, ADR-012)** |
| Null semantics (pandas-flavored) | yes | yes |
| `filter` / `groupby` / `join` / `sortValues` | yes | yes |
| `withColumn` / `select` / `drop` / `head` / `tail` | yes | yes |
| `describe()` | yes | yes |
| Null-propagating arithmetic + Kleene boolean | yes | yes |
| Expression compiler + kernel fusion | yes | yes |
| Lambda escape hatch (`filterFn` / `mapFn`) | yes | yes |
| SIMD kernels (auto-detected) | yes | yes |
| Opt-in worker threads (`enableThreads()`) | yes | yes |
| CSV / JSON / Arrow IPC I/O | yes | yes |
| Parquet I/O (`databonk/parquet` subpath) | no | **yes (ADR-011)** |
| Chunked / out-of-core columns | no | planned |
| Lazy query optimizer | no | planned |
| `pandas.Index` / alignment | no | not planned |
| Mutation-in-place API | no | not planned |
| wasm64 (> 4 GB) | no | consideration |

---

## Parallel / threaded mode

Requires `SharedArrayBuffer`. In Node ≥ 18 this is always available. In the
browser you must enable cross-origin isolation (COOP/COEP headers).

```typescript
import { enableThreads } from 'databonk/workers';

const th = await enableThreads({ workers: 4 });
if (!th) throw new Error('threads unavailable — check COOP/COEP headers');

// th.sumF64 / th.meanF64 / th.minF64 / th.maxF64 dispatch in parallel
const sum = await th.sumF64(dataPtr, vpPtr, len);
th.terminate();
```

Full docs: [docs/threads.md](docs/threads.md)

---

## Bundler / WASM loading

See [docs/bundlers.md](docs/bundlers.md) for Vite, webpack, and inline-base64
fallback instructions.

---

## API docs

Generated with [TypeDoc](https://typedoc.org). Run `npm run docs` to build
`docs/api/` from source. The generated output is not committed to the repository.

---

## Development

The build toolchain runs inside Docker (Rust + wasm-opt + Node):

```bash
# Build wasm + JS
docker run --rm -v "$PWD":/work -w /work dataframe-dev \
  bash -lc 'npm ci && npm run gate'

# Run tests only
docker run --rm -v "$PWD":/work -w /work dataframe-dev \
  bash -lc 'npm ci && npm run test'

# Run E2E benchmarks
docker run --rm -v "$PWD":/work -w /work dataframe-dev \
  bash -lc 'npm ci && npm run build && node bench/e2e/pipeline.mjs'
```

Issue tracking: **bd (beads)** — `bd ready` to find available work, `bd prime`
for full workflow context.

---

## License

MIT
