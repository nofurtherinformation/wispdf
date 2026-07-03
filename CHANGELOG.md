# Changelog

All notable changes are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Future releases managed via [Changesets](https://github.com/changesets/changesets).

---

## [0.2.0] — 2026-07-03

v0.2.0 reverses three explicit v1 non-goals: **i64/BigInt columns**, **temporal
dtypes**, and **Parquet I/O**. It also raises the JS entry size budget from 25 KB
to 30 KB (ADR-012) to accommodate the enlarged surface. Consumers who import only
the v1 profile through a tree-shaking bundler continue to pay approximately v1 cost.

### v2.1 — Contracts: ADR-009/010/011 + v2 dtype/ABI deltas

- ADR-009 (i64): `BigInt64Array` zero-copy storage; safe-int throw on unsafe `number`
  literals; wrapping two's-complement arithmetic; `sum/min/max` return `bigint`;
  `mean/std/var` return `f64` with documented precision caveat above 2^53.
- ADR-010 (temporals): `date32` (i32 days since epoch) + `timestamp` (i64 UTC ms);
  tz as column metadata only (no wasm; reuses i32/i64 kernels); ISO 8601 weekday
  numbering (Mon=1…Sun=7); `dt` accessor proxy over the typed-array view.
- ADR-011 (parquet): `databonk/parquet` subpath; runtime deps (`hyparquet` +
  `hyparquet-writer`) confined to the subpath; main entry stays dep-free ≤ 30 KB.
- `contracts/dtypes.md` §6–§12 (full v2 cast matrix, temporal-arith restriction table,
  dt accessors); `contracts/wasm-abi.md` §10–§11; `contracts/memory.d.ts` extended.
- ADR-012: JS entry budget raised to **30 KB gz** (from 25 KB); wasm budgets unchanged.

### v2.2 — Conformance fixtures (i64 + temporal + parquet)

- 19 i64 conformance cases (wrapping overflow, safe-int throw, `f64→i64` out-of-range→null,
  `i64→i32/u32` wrap, div/mod zero→null, widening lattice).
- 31 temporal cases (negative `timestamp→date32` floor, ISO weekday across a week including
  pre-epoch, tz-aware vs UTC accessor across a DST boundary, all arithmetic-restriction errors).
- 19 parquet cases (12 round-trip, 6 out-of-profile error, 1 smoke 1M-row).

### v2.3 — i64 wasm kernels

- `add/sub/mul/neg/div/mod_i64`, six comparisons, `filter/gather/argsort/topk_i64`,
  `min/max/first/last/sum/mean/std/var/nunique_i64`, `hash_i64` (splitmix64).
- SIMD build uses `i64x2` for add/sub/neg/comparisons; mul/div/mod are scalar-only in
  both builds (SIMD128 provides no integer-division instruction).
- Gate: WASM `add_i64` ≥ 1.5× JS `BigInt64Array` loop at 1M elements. **Actual: 2.91×.**

### v2.4 — i64 JS layer

- `BigInt64Array` column path; `bigint` boundary at all scalar crossings.
- Widening lattice in the expression compiler (i32/u32 → i64; i64 → f64; i64 ⊕ f32 → f64).
- i64 reductions return `bigint`; `mean/std/var_i64` return `f64`.
- Safe-int literal throw for `number` values outside `Number.isSafeInteger` range.

### v2.5 — Temporal layer: date32 / timestamp / dt accessors / tz metadata

- `date32` and `timestamp` dtypes added to the dtype registry; physical kernel tokens
  (`date32` → `i32`, `timestamp` → `i64`) reuse all existing kernels.
- Restricted arithmetic: `timestamp − timestamp → i64 (ms)`;
  `timestamp ± integer-ms → timestamp`; `date32 − date32 → i32 (days)`;
  `date32 ± integer-days → date32`; all other temporal arithmetic is a typed error.
- `dt` accessor proxy (`col('ts').dt.year()`, `.month()`, `.day()`, `.hour()`,
  `.minute()`, `.second()`, `.millisecond()`, `.weekday()`, `.dayOfYear()`, `.quarter()`).
  UTC path: integer civil-from-days math (Howard Hinnant). tz-aware path: cached
  `Intl.DateTimeFormat` (no shipped tz database).
- `Date` objects accepted in column construction (`date32`: → days; `timestamp`: → ms).
- `tzs` option on `DataFrame.fromColumns` to attach IANA tz metadata.
- `Column.tz` field; `Series.dt` accessor proxy.
- ADR-012: `dt` accessors activate fully after the budget raise (they shipped as
  throwing stubs in the i64+temporal integration pass due to the 25 KB gate).

### v2.6 — Arrow/CSV i64 + temporal integration

- Arrow write: `i64 → Int64`, `date32 → Date[DAY]`, `timestamp → Timestamp[MILLI, tz]`.
- Arrow read: `Int64 → i64`, `Date[DAY] → date32`, `Timestamp[MILLI, tz] → timestamp+tz`;
  SECOND/MICRO/NANO rescaled to ms (saturation-to-null on overflow).
- CSV: automatic i64 inference for integers > `Number.MAX_SAFE_INTEGER`; explicit
  `dtypes: { col: 'i64' | 'date32' | 'timestamp' }` parse paths.
- 1099 tests green (scalar + SIMD builds); main entry 25.4 KB gz (≤ 30 KB gate).

### v2.7 — Parquet subpath (`databonk/parquet`)

- `readParquet(bytes, rt)` (async) + `writeParquet(df, opts?)` (sync) for all 9
  supported dtypes: `f64 f32 i32 u32 i64 bool utf8 date32 timestamp`.
- Snappy + uncompressed; null validity; dictionary-encoded `utf8`; `timestamp` tz
  round-trips in file `key_value_metadata` (`databonk:tz:<colname>`).
- Out-of-profile inputs (gzip/zstd/brotli/lz4, INT96, DECIMAL, nested/repeated,
  `FIXED_LEN_BYTE_ARRAY`) raise a clear, specific "unsupported" error.
- Runtime deps (`hyparquet 1.26.2` + `hyparquet-writer 0.16.1`) confined to the
  subpath; main entry not touched. Subpath ≈ 20.9 KB gz (databonk adapter +
  hyparquet/hyparquet-writer as external imports; not bundled into the tarball).
- `parquet-wasm 0.7.2` as a devDep test oracle; never shipped.

### v2.8 — Release 0.2.0 (this release)

- Version bumped to **0.2.0**; `src/index.ts` `VERSION` constant updated.
- CHANGELOG, README, examples, typedoc, and docs/status.md updated for v2 surface.
- Fresh-clone gate (npm ci + npm run gate + gate:bench + check:readme) all green.

### Gate results (0.2.0)

| Gate | Result |
|---|---|
| 1164 tests (scalar + SIMD builds) | PASS |
| JS entry ≤ 30 KB gz (ADR-012) | 27.0 KB — PASS |
| Each wasm ≤ 75 KB gz | 17.4 / 21.9 / 22.1 KB — PASS |
| E2E pipeline ≥ 1× Arquero at 1M rows | 3.80× — PASS |
| i64 add_i64 ≥ 1.5× JS BigInt64Array at 1M | 2.91× — PASS |
| v1 regression gate (≤ 10% on 5 ops) | PASS (pipeline 21% faster) |
| publint | PASS |
| arethetypeswrong | PASS |
| npm publish --dry-run | PASS |

---

## [0.1.0] — 2026-07-02

First public release of **databonk** — a columnar WASM dataframe library for JavaScript.

### Phase 0 — Foundation & language spike

- Repository scaffold: TypeScript strict, tsup (dual ESM/CJS + `.d.ts`), vitest, tinybench, GitHub Actions CI.
- Benchmark harness and dataset generators (100K / 1M / 10M rows). Baselines recorded for typed-array JS, array-of-objects JS, Arquero, and Danfo.js.
- Language spike comparing AssemblyScript vs Rust for three kernel families (add, null-aware sum, comparison-to-bitmask). **Rust selected** (ADR-007) based on throughput and binary size.
- `contracts/wasm-abi.md` v1 and `contracts/dtypes.md` written.

### Phase 1 — Memory core

- Rust bump+freelist arena allocator with `alloc`/`free`/`realloc`/`mem_generation` exports.
- Dual scalar/SIMD wasm builds; feature-detected loader via `WebAssembly.validate`.
- Column representation: contiguous data buffer + Arrow-compatible validity bitmap (LSB bit-pack).
- Dictionary string store: build, unify two dictionaries, decode-memoization on the JS side.
- `viewOf()` with generation-counter invalidation (ADR-001: `memory.grow` safety).
- `contracts/memory.d.ts` v1 finalized.

### Phase 2 — Kernels (parallel fan-out, 4 agents)

- **Elementwise:** f64/f32/i32/u32/bool arithmetic (add/sub/mul/div/mod/neg), comparisons (→ validity bitmask), Kleene boolean, cast, fillNull.
- **Reductions:** sum/mean/min/max/count/nunique/std/var/first/last; null-aware; SIMD-accelerated where profitable.
- **Selection:** filter-by-bitmask (compaction), gather/take, stable argsort (merge sort with scratch buffer, O(n log n)), top-k, zero-copy slice.
- **Relational:** 64-bit column hashing, hash-based groupby, hash join (inner + left), dictionary unification hook.
- ABI v1.2 amendment: `argsort_dt` gained caller-provided scratch pointer for stable O(n log n) merge sort.

### Phase 3 — Expression & API layer

- Expression AST + compiler (`col`, `lit`, arithmetic, comparison, boolean, null, cast, aggregation expressions).
- Kernel-call fusion: `compare → filter` emits one mask + one compaction; chained elementwise ops reuse one output buffer.
- `DataFrame` / `Series` / `GroupBy` / `join` implementing the full §4 API surface.
- Lambda escape hatch: `filterFn` / `mapFn` iterate via a reusable row-proxy over `viewOf()` views (zero-copy, scalar JS speed — documented slow path per ADR-003).
- Reference-counted buffer sharing (`OwnedColumn`); `dispose()` / `scope()` for WASM memory lifecycle.
- Helpful error messages (unknown column → nearest-match suggestion; dtype mismatch → both types named).
- Table pretty-printer for `console.log` / `toString()`.

### Phase 4 — Hardening & performance sweep

- Minified build (17.1 KB gz ESM at P4 baseline; 22.4 KB at P6 after I/O).
- `WASM_BUILD` env-var test split (scalar / SIMD builds verified independently).
- Fuzz suite (fast-check) covering the full public API surface.
- Regression harness (`bench/baselines/wasm-v1.json` + `scripts/check-regression.mjs`); CI fails on > 10% regression.

### Phase 5 — Parallel mode

- `simd-threads.wasm` build (nightly Rust, `+atomics +bulk-memory`, imported `SharedArrayBuffer`-backed memory).
- `enableThreads({ workers })` — exported from `databonk/workers` subpath entry (separate bundle to keep main entry under 25 KB gate).
- Chunk dispatch: elementwise ops write directly into shared memory; reductions combine partial sums left-to-right (deterministic, non-bit-identical to single-thread for f64 — documented deviation).
- Worker crash/timeout recovery: failed workers are terminated and respawned automatically.
- Graceful no-op when `crossOriginIsolated` is absent (browser without COOP/COEP).
- Gate: ≥ 1.8× speedup on 4 workers for 10M-row reductions. Actual: 3.3–3.5×.

### Phase 6 — I/O, release

- **I/O (Agent E):** CSV reader (type inference, streaming-friendly), JSON records (`fromJSON`/`toJSON`), Arrow IPC (`fromArrow`/`toArrow`) verified against `apache-arrow` (dev-only dep; no runtime dep).
- **Release (Agent F — this release):**
  - Package name `databonk`, version `0.1.0`, license MIT, `sideEffects: false`.
  - Exports map with split `import`/`require` + per-condition `.d.ts`/`.d.cts` types (attw-clean).
  - `README.md` with honest benchmark table, feature matrix, quickstart, bundler section.
  - `docs/bundlers.md` — Vite, webpack, Node, inline-base64 fallback.
  - TypeDoc config (`typedoc.json`); `npm run docs` generates `docs/api/`.
  - `examples/node-quickstart/` — plain Node script exercising the README quickstart against the built package.
  - `examples/vite-app/` — minimal Vite page doing a 100K-row pipeline; `vite build` verified green.
  - publint + `@arethetypeswrong/cli` packaging checks clean.
  - `npm publish --dry-run` clean.
  - Changesets config for future releases.

### Gate results (0.1.0)

| Gate | Result |
|---|---|
| 877 tests (scalar + SIMD builds) | PASS |
| JS entry ≤ 25 KB gz | 22.4 KB — PASS |
| Each wasm ≤ 75 KB gz | 15.0 / 18.4 / 18.6 KB — PASS |
| E2E pipeline ≥ 1× Arquero at 1M rows | 3.8× — PASS |
| Threads ≥ 1.8× on 4 workers, 10M rows | 3.3–3.5× — PASS |
| publint | PASS |
| arethetypeswrong | PASS |
| npm publish --dry-run | PASS |

---

[0.1.0]: https://github.com/TODO/databonk/releases/tag/v0.1.0
