# Phase 0 Status Ledger

Maintained by the orchestrator (Fable). Subagents must not edit this file.

Last updated: 2026-07-02

## Task Registry

| Bead ID | Task | Agent | Status | Gate Result | Notes |
|---|---|---|---|---|---|
| dataframe-99s.1 | P0.1 Repo scaffold + tooling + ADR transcription | Sonnet | in-progress | — | — |
| dataframe-99s.2 | P0.2 Bench harness + dataset generators + 4 JS baselines | — | blocked (needs .1) | — | — |
| dataframe-99s.3 | P0.3a Language spike: AssemblyScript kernels | — | blocked (needs .1) | — | — |
| dataframe-99s.4 | P0.3b Language spike: Rust kernels | — | blocked (needs .1) | — | — |
| dataframe-99s.5 | P0.4 Contracts: wasm-abi.md + dtypes.md | — | blocked (needs .3/.4) | — | — |

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
| ADR-007 | Implementation language: decided by Phase 0 spike | pending |
| ADR-008 | Stable kernel ABI | accepted |
