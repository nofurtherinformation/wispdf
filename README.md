# skidi

Columnar dataframe library for JavaScript — pandas-familiar API, WebAssembly-accelerated kernels, zero-copy typed-array views, dual ESM/CJS, TypeScript types.

```
npm install skidi
```

Works in **Node.js ≥ 18**, **Vite**, and **webpack** without configuration gymnastics.
Each `.wasm` binary is ≤ 19 KB gzipped; the JS entry is ≤ 23 KB gzipped.

---

## Why skidi?

Most JS "dataframe" libraries either wrap pandas in WASM (giant binary) or operate
row-by-row over plain objects (slow).  skidi takes a different path:

- **Columns live in WASM linear memory.** JS holds zero-copy `TypedArray` views over
  those buffers — no marshalling overhead on the hot path.
- **SIMD kernels** (auto-detected; scalar fallback for older Safari). Core
  operations are 3–4× faster than Arquero on 1M-row pipelines.
- **pandas-shaped API** (`filter`, `groupby().agg()`, `sortValues`, `join`) without
  pandas' index-alignment surprises.
- **Opt-in threads** via `enableThreads()` (COOP/COEP required in browser; always
  available in Node). 3.3–3.5× speedup on 4 workers for 10M-row reductions.
- **I/O** — CSV, JSON records, Arrow IPC in/out (no runtime Arrow dependency).

### v1 non-goals (honest)

The following are deferred to v2 or not planned:

- `pandas.Index` / automatic alignment (row position is identity)
- `i64` / BigInt columns; dates, timestamps, timezones
- Chunked / out-of-core columns; lazy query optimizer
- Parquet I/O (Arrow IPC only); write-side CSV formatting
- Mutation-in-place API (all ops return new frames)

---

## Install

```
npm install skidi
```

Peer-required: **Node.js ≥ 18** (or a modern browser with WebAssembly support).

---

## Quickstart

```typescript
import { init, DataFrame, col } from 'skidi';

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
import { scope } from 'skidi';

const result = scope(() => {
  const filtered = df.filter(col('value').gt(5));
  const grouped  = filtered.groupby(['group']).agg({ value: 'sum' });
  return grouped.toRecords();   // primitive — safe to return outside scope
});
// all intermediate frames disposed automatically
```

### I/O

```typescript
import { fromCSV, fromArrow, toArrow, fromJSON, toJSON } from 'skidi';

// CSV (auto-infers dtypes)
const df = await fromCSV(csvText, { delimiter: ',', hasHeader: true });

// Arrow IPC (compatible with Apache Arrow; no runtime arrow dep needed)
const buf = toArrow(df);
const df2 = fromArrow(buf);

// JSON records
const df3 = fromJSON([{ x: 1, y: 'a' }, { x: 2, y: 'b' }]);
const json = toJSON(df3);
```

---

## Expression API vs lambda escape hatch

skidi has two filter/map styles:

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

Measured 2026-07-02 on **Node 22, Apple M-series** (single thread, SIMD build).
Numbers are median wall-clock milliseconds over 7 runs at 1M rows.

| Operation | skidi (ms) | Arquero (ms) | Ratio |
|---|---:|---:|---:|
| filter → groupby → sum (pipeline) | 11.9 | 45.5 | **3.8× faster** |
| join (inner, string key) | 22.4 | 38.9 | **1.7× faster** |
| sortValues (f64, 1M rows) | 93.2 | 155.6 | **1.7× faster** |

> **Caveats:** Arquero times include `.objects()` materialisation (that's what its
> pipeline naturally produces); skidi times do not materialise to JS objects.
> Results vary by machine, Node version, dataset shape, and JIT warm-up.
> Run `node bench/e2e/pipeline.mjs` after `npm run build` to reproduce.

Danfo.js is tracked in `/bench/baselines/danfo.json` for informational comparison
but is not shown here (it uses TensorFlow.js under the hood, making size
comparisons misleading).

### Parallel mode (4 workers, 10M f64 elements)

| Op | 1 thread (ms) | 4 workers (ms) | Speedup |
|---|---:|---:|---:|
| sum | 2.74 | 0.82 | 3.3× |
| mean | 2.74 | 0.82 | 3.4× |
| min | 4.38 | 1.24 | 3.5× |

See [docs/threads.md](docs/threads.md) for setup.

---

## Bundle sizes

Gzipped, 0.1.0 build (Node 22, Apple M-series, 2026-07-02):

| Asset | Gzipped |
|---|---:|
| `dist/index.js` (ESM) | 22.4 KB |
| `dist/index.cjs` (CJS) | 22.7 KB |
| `dist/simd.wasm` | 18.6 KB |
| `dist/scalar.wasm` (fallback) | 15.0 KB |
| `dist/simd-threads.wasm` (threads opt-in) | 18.4 KB |

---

## Feature matrix

| Feature | v0.1.0 | v2 target |
|---|---|---|
| f64 / f32 / i32 / u32 / bool / utf8 columns | yes | — |
| i64 / BigInt columns | no | planned |
| Date / timestamp / timezone | no | planned |
| Null semantics (pandas-flavored) | yes | — |
| `filter` / `groupby` / `join` / `sortValues` | yes | — |
| `withColumn` / `select` / `drop` / `head` / `tail` | yes | — |
| `describe()` | yes | — |
| Null-propagating arithmetic + Kleene boolean | yes | — |
| Expression compiler + kernel fusion | yes | — |
| Lambda escape hatch (`filterFn` / `mapFn`) | yes | — |
| SIMD kernels (auto-detected) | yes | — |
| Opt-in worker threads (`enableThreads()`) | yes | — |
| CSV / JSON / Arrow IPC I/O | yes | — |
| Parquet I/O | no | planned |
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
import { enableThreads } from 'skidi/workers';

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
