# Status Ledger (Phase 0 → Phase 3)

Maintained by the orchestrator (Fable). Subagents must not edit this file (the v2
lead seeds the v2 ledger below at commission; downstream v2 agents append notes via
`bd`, not by editing this file).

Last updated: 2026-07-03 (v0.2.0 shipped)

## Task Registry

| Bead ID | Task | Agent | Status | Gate Result | Notes |
|---|---|---|---|---|---|
| dataframe-99s.1 | P0.1 Repo scaffold + tooling + ADR transcription | Sonnet | done | `npm run gate` green (build+test+size) | 8 ADRs transcribed; CI/size gates on hello-world module |
| dataframe-99s.2 | P0.2 Bench harness + dataset generators + 4 JS baselines | Sonnet | done | baselines recorded | typedarray/arrayobj/arquero/danfo JSON in `bench/baselines/` |
| dataframe-99s.3 | P0.3a Language spike: AssemblyScript kernels | Sonnet | done | correctness PASS (both builds) | 3 kernels × {scalar,simd}; gzip 500/582 B |
| dataframe-99s.4 | P0.3b Language spike: Rust kernels | Sonnet | done | correctness PASS after LEAD fix | 3 kernels × {scalar,simd}; gzip 673/910 B. **LEAD fixed a SIMD `sum_f64_null` null-path double-count bug** (splat into both lanes); fast-path throughput unchanged. |
| dataframe-99s.5 | P0.4 ADR-007 decision + contracts (wasm-abi.md + dtypes.md) | Opus (LEAD) | done | fresh benches re-run + verified; `npm run gate` green | **ADR-007 = Rust** (accepted). `contracts/wasm-abi.md` v1 + `contracts/dtypes.md` v1 written. |
| dataframe-39z.1 | P1.1 Rust arena allocator + wasm build infra + viewOf layer | Opus | done | `npm run gate` green (build+test+size); 29 tests | Bump+freelist arena (`alloc`/`free`/`realloc`/`mem_generation`), dual scalar/simd builds, SIMD-detect loader, single `viewOf` generation-counter accessor. `contracts/memory.d.ts` v0.9-draft. |
| dataframe-39z.2 | P1.2 Columns, dict string store, dtype registry, zero-copy slice | Opus | done | `npm run gate` green (build+test+size); 50 tests | Dtype registry + Arrow validity bitmap + column create/toArray (typed fast + null-detecting slow path) + dict store (build/decode-memo/unify) + zero-copy slice (data byte-offset baked in `dataPtr`, validity bit-offset). `contracts/memory.d.ts` **v1 final**. |
| (P2 bench triage) | ABI **v1.2** amendment: `argsort_dt` gains caller `scratch_ptr` (restores O(n log n) stable merge; no-scratch rotation-merge missed the §5 gate); `filter_indices` kept exported but JS dispatch wins (V8 ctz beats wasm). | Opus (LEAD) | done | commit 8319028 | Contract-only fix; kernel `scratch_ptr` landing separately. **NB:** the argsort binary in this worktree is still pre-amendment (arity-5) → `sortValues` is slow until the amended kernel merges. |
| dataframe-9qm.1 | P3.1 Expression AST + compiler + fusion | Opus | done | `npm run gate` green; 719 tests (75 new) | `compile`/`compileFilter` over `FrameView`; compare→filter + elementwise-chain fusion verified by `ExecStats`; JS entry 13.56 KB gz. commit e1d27d9. |
| dataframe-9qm.2 | P3.2 DataFrame/Series/GroupBy/join API + row proxy + errors + printer | Opus | done | `npm run gate` green (build+test+size); **768 tests (49 new)**; size OK | `src/frame/**` + public exports. E2E **pipeline 3.83× Arquero** (11.9 ms vs 45.5 ms @1M SIMD; gate ≥1×). join (inner) 1.73× Arquero. `sortValues` 0.02× — **blocked on the pre-v1.2 slow `argsort` kernel** (coded against the stable API with scratch-arity detection, inherits O(n log n) once it lands). JS entry 23.78 KB gz. Ref-counted buffer sharing + `dispose()`/`scope()`; frame leak test green. |
| dataframe-8aj.1 | P4.1 Hardening & performance sweep | Sonnet | in-progress | `npm run gate` green (build+test:scalar+test:simd+size); **771 tests** both builds; index.js 17.13 KB gz, index.cjs 17.44 KB gz | minify ON; WASM_BUILD env-var scalar/simd test split; fuzz suite (3 new tests); regression harness (wasm-v1.json + check-regression.mjs); bench cleanup. `gate:bench` NOT included in `gate` script. |
| dataframe-8aj.2 | P4.2 Independent adversarial verification | Sonnet | done | Full gate + bench matrix re-run green; regression harness injection PASS; 2 findings | See bead notes. **Findings:** (1) `withColumn_add_100k` baseline (0.0953 ms) flaky: 2/3 fresh Docker runs fail at 1.11× (threshold 1.10); sub-ms op is noise-sensitive. (2) `--update` has no dirty-tree guard (noted, not fixed). E2E reproduced: pipeline 3.72-3.79×, join 1.74-1.79×, sortValues 1.68-1.70×. 0 skipped tests; test diff additive only. |

| dataframe-5am.2 | P6.F Release prep: README, typedoc, examples, packaging checks, publish dry-run | Sonnet | done | gate 877/877; publint clean; attw all-green (node10/node16/bundler); publish dry-run clean; node-quickstart PASS; vite build PASS; typedoc 1.7 MB; pack 385.7 KB / 19 files | CSV whitespace-as-f64 bug fixed; worker-script publint false-positive fixed via `require` alias; `typesVersions` added for node10 subpath; `sideEffects:false`; `private` removed; exports map split per import/require condition. |

## v2 Ledger (Epic dataframe-dh9 — i64 / temporals / Parquet, target 0.2.0)

Commissioned by Dylan 2026-07-02. Each of i64, temporals, and Parquet reverses a
documented v1 non-goal, so each carries an ADR (ADR-009/010/011). Contract v2 deltas live
in `contracts/dtypes.md` §6–§12, `contracts/wasm-abi.md` §10–§11, and `contracts/memory.d.ts`.

| Bead ID | Task | Agent | Status | Depends on | Notes |
|---|---|---|---|---|---|
| dataframe-dh9.1 | v2.1 Contracts: ADR-009/010/011 + dtypes/wasm-abi/memory v2 deltas | v2 LEAD (Opus) | in-progress | — | ADR-009 (i64), ADR-010 (temporals), ADR-011 (parquet) accepted; `dtypes.md` §6–§12 (dtype rows, full v2 cast matrix incl. i64→f64 round / f64→i64 null / temporal reinterpret+scale w/ negative floor-div example, widening lattice, temporal-arith restriction, dt accessors ISO Mon=1..Sun=7); `wasm-abi.md` §10–§11 (i64 export list all 4 families + BigInt64 crossing + i64 reduction scratch/identity table + temporal reuse mapping, NO temporal wasm exports); `memory.d.ts` extended (tsc-clean). ADR-011 sizes measured (reader+writer subpath ≈29 KB gz). |
| dataframe-dh9.2 | v2.2 Conformance fixtures: i64 + temporal cases (author/verify/fix) | TBD | open | dh9.1 | Must cover: wrapping overflow, i64→f64 >2^53 rounding, f64→i64 range→null, i64→i32/u32 wrap, div/mod zero→null, safe-int literal throw, negative timestamp→date32 floor, ISO weekday across a week (incl. pre-epoch), tz vs UTC accessor across DST, arith-restriction errors, Parquet in/out-of-profile. |
| dataframe-dh9.3 | v2.3 i64 wasm kernels across all 4 families | TBD | blocked | dh9.1 | Implement `wasm-abi.md` §10 exports (both builds); SIMD `i64x2` for add/sub/neg/cmp, scalar mul/div/mod (§10.5); `hash_i64` = splitmix64. |
| dataframe-dh9.4 | v2.4 i64 JS layer: registry, column, expr, frame | TBD | blocked | dh9.3 | `BigInt64Array` column path; bigint boundary; widening lattice in the expr compiler; i64 reductions return bigint. |
| dataframe-dh9.5 | v2.5 Temporal layer: date32/timestamp, dt accessors, tz metadata | TBD | blocked | dh9.4 | Registry logical→physical token; restricted temporal algebra; JS-side dt accessors (civil-from-days + cached `Intl.DateTimeFormat`); scale-casts; `Column.tz`. No new wasm. |
| dataframe-dh9.6 | v2.6 Arrow/CSV integration for i64+temporals | Sonnet | done | dh9.5 | Arrow write: i64→Int64, date32→Date[DAY], timestamp→Timestamp[MILLI,tz]. Arrow read: Int64/Date32/Timestamp→dtypes; SECOND/MICRO/NANO rescale to ms with saturation-to-null on overflow. CSV: i64 inference (integer >MAX_SAFE_INTEGER promotes to i64); explicit `dtypes:{col:'i64'|'date32'|'timestamp'}` parse paths; reject ambiguous local-time for timestamp. **1099 tests green (scalar+simd); index.js 25.4 KB gz (≤30 KB gate).** Merged. |
| dataframe-dh9.7 | v2.7 Parquet subpath: `databonk/parquet` reader+writer | Sonnet | done | dh9.5 | `src/parquet/index.ts` — `readParquet`+`writeParquet` full ADR-011 profile (9 dtypes, snappy+uncompressed, null validity, tz round-trip via key_value_metadata). tsup entry + package.json exports map + typesVersions updated. 19 conformance tests (12 round-trip, 6 error cases, 1 smoke 1M-row). Main entry 24.46 KB gz (≤30 KB). publint clean; attw green. Deps: hyparquet 1.26.2 + hyparquet-writer 0.16.1 (exact-pinned); parquet-wasm 0.7.2 devDep. Merged. |
| dataframe-dh9.8 | v2.8 Release 0.2.0: docs, CHANGELOG, fresh-clone verify | Sonnet | done | gate+gate:bench+check:readme green; publint+attw+dry-run clean; node-quickstart+vite-app green; 27.0 KB gz entry | package.json 0.2.0; CHANGELOG 0.2.0; README v2 sections (i64, temporals, parquet, updated matrix + sizes); check-readme.mjs +parquet path; examples/node-quickstart i64+temporal+parquet; docs/status.md closed. Pack: 27 files, 700.9 KB, 2.9 MB unpacked. hyparquet/hyparquet-writer are external imports (not bundled into tarball). |

## Gate Definitions

| Gate | Condition | Blocks |
|---|---|---|
| size-js | gzipped dist/index.js ≤ 30 KB (ADR-012; was 25 KB for v1 profile) | all phases |
| size-wasm | each gzipped *.wasm ≤ 75 KB | P2+ |
| test | vitest suite green | all phases |
| bench-kernel | SIMD kernel ≥ 1.5× typed-array baseline at 1M rows | P2 merge |
| bench-e2e | filter→groupby→agg at 1M rows ≥ Arquero | P3 merge |
| bench-regression | no > 10% regression vs stored baseline JSON | P2+ CI |

## ADR Index

| ADR | Title | Status |
|---|---|---|
| ADR-001 | WASM-core memory ownership | accepted |
| ADR-002 | Arrow-compatible columnar layout | accepted |
| ADR-003 | Hybrid query API | accepted |
| ADR-004 | Dual WASM builds, feature-detected | accepted |
| ADR-005 | No index; hash-based relational ops | accepted |
| ADR-006 | Parallelism is an opt-in shared-memory mode | accepted |
| ADR-007 | Implementation language: **Rust** (decided by Phase 0 spike) | accepted |
| ADR-008 | Stable kernel ABI | accepted |
| ADR-009 | i64 / BigInt columns (reverses v1 non-goal) | accepted |
| ADR-010 | Temporal dtypes: date32, timestamp, tz metadata (reverses v1 non-goal) | accepted |
| ADR-011 | Parquet I/O via a scoped `databonk/parquet` subpath (reverses v1 non-goal) | accepted |
| ADR-012 | JS entry size budget raised to 30 KB gz for v2 surface | accepted |

## v2 Parking Lot

Features remaining after v0.2.0 release. Items shipped in v0.2.0 removed from this list.

| Feature | Notes / Blocker |
|---|---|
| Chunked / out-of-core columns | Requires Arrow Chunked Array layout change; lazy optimizer prerequisite |
| Lazy query optimizer | Requires logical plan representation; needed for predicate push-down and projection pruning |
| wasm64 (> 4 GB memory) | Requires wasm64 target support in Rust + browser runtime support |
| `utf8` col-vs-col compare | Kernel ABI requires unify_dict before comparison; deferred from v2 scope |
| `nunique` O(n²) ceiling | Current hash-table implementation; documented acceptable limit at ≤ 1M unique values |
| ReadableStream / streaming CSV | `ChunkParser` is streaming-friendly internally; `fromCSVStream(stream)` deferred — usage pattern unclear |
| pandas.Index / alignment | Explicit non-goal; row position is identity |
| Mutation-in-place API | Explicit non-goal; all ops return new frames |
