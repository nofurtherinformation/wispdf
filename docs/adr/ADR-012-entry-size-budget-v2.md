# ADR-012 — JS entry size budget raised to 30 KB gzipped for the v2 surface

Status: accepted (orchestrator, 2026-07-02). Supersedes the 25 KB figure in spec §1/§5 for v2+.

## Context
The v1 budget (25 KB gz) was set for the v1 feature profile and was met with 7.6 KB of
headroom (17.4 KB after minification). Dylan commissioned three v2 features that were
explicit v1 non-goals: i64/BigInt (ADR-009), temporal dtypes (ADR-010), Parquet
(ADR-011). i64 + temporal integration alone brought the entry to 24.76 KB — leaving 3
bytes of headroom and forcing the temporal `dt` accessors (a core deliverable of
ADR-010) to ship as throwing stubs.

## Decision
- Main-entry budget: **30 KB gzipped** (`scripts/check-size.mjs`). Wasm budgets unchanged (75 KB each).
- Parquet stays out of the main entry entirely (`databonk/parquet` subpath, ADR-011), as do threads (`databonk/workers`).
- `sideEffects:false` + ESM mean tree-shaking consumers who import the v1 profile pay ~v1 cost; the gate measures the worst-case full import.

## Consequences
`dt` accessors activate fully (evalDt + Series.dt incl. tz variants). Future surface
growth must fit 30 KB or ship as a subpath; the next raise requires a new ADR with
numbers, same as this one.

## Alternatives rejected
- Temporal accessors as a subpath: fractures the expression API (`col('t').dt.year()` is core ergonomics, not an optional capability).
- Aggressive code-golf of frame/expr: bought ~0.3 KB in trials during P6.F; not renewable.
