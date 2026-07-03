# Changelog

All notable changes are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Future releases managed via [Changesets](https://github.com/changesets/changesets).

---

## [0.1.0] — 2026-07-02

First public release of **wispdf** — a columnar WASM dataframe library for JavaScript.

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
- `enableThreads({ workers })` — exported from `wispdf/workers` subpath entry (separate bundle to keep main entry under 25 KB gate).
- Chunk dispatch: elementwise ops write directly into shared memory; reductions combine partial sums left-to-right (deterministic, non-bit-identical to single-thread for f64 — documented deviation).
- Worker crash/timeout recovery: failed workers are terminated and respawned automatically.
- Graceful no-op when `crossOriginIsolated` is absent (browser without COOP/COEP).
- Gate: ≥ 1.8× speedup on 4 workers for 10M-row reductions. Actual: 3.3–3.5×.

### Phase 6 — I/O, release

- **I/O (Agent E):** CSV reader (type inference, streaming-friendly), JSON records (`fromJSON`/`toJSON`), Arrow IPC (`fromArrow`/`toArrow`) verified against `apache-arrow` (dev-only dep; no runtime dep).
- **Release (Agent F — this release):**
  - Package name `wispdf`, version `0.1.0`, license MIT, `sideEffects: false`.
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

[0.1.0]: https://github.com/TODO/wispdf/releases/tag/v0.1.0
