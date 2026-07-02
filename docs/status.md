# Status Ledger (Phase 0 → Phase 1)

Maintained by the orchestrator (Fable). Subagents must not edit this file.

Last updated: 2026-07-02

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

## Gate Definitions

| Gate | Condition | Blocks |
|---|---|---|
| size-js | gzipped dist/index.js ≤ 25 KB | all phases |
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
