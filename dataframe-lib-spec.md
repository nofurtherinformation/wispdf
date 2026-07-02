# Spec: Columnar DataFrame Library for JavaScript (working name: TBD)

**Handoff document for autonomous multi-agent build.**
Orchestrator: Claude Fable. Subagents: Claude Sonnet (default) / Opus (foundational phases).
Status: v1.0 — decisions below are locked unless an ADR supersedes them.

---

## 0. Summary

A lightweight, npm-installable, columnar dataframe library for browser and Node with
pandas-familiar ergonomics. Columns live in WebAssembly linear memory ("WASM-core");
JavaScript holds zero-copy typed-array views over that memory. Vectorized (SIMD)
kernels execute in WASM; a hybrid API offers a fast expression path plus a JS lambda
escape hatch that iterates the same views without copying.

## 1. Design goals

1. **Lightweight.** JS entry ≤ 25 KB gzipped; each wasm binary ≤ 75 KB gzipped, lazy-loaded.
2. **Fast.** Beats idiomatic JS loops (both array-of-objects and typed-array baselines) on core columnar ops; matches or beats Arquero on end-to-end pipelines at 1M rows.
3. **Ergonomic.** pandas-familiar naming and shape (`filter`, `groupby().agg()`, `sortValues`, `head`) without pandas' index-alignment semantics.
4. **Installable.** `npm install <name>`; dual ESM/CJS, TypeScript types, works in Vite/webpack/Node ≥ 18 without config gymnastics.

### Non-goals (v1)

- pandas Index / automatic alignment (row position is identity)
- i64 / BigInt columns, dates/timestamps/timezones (v2)
- Chunked columns, out-of-core, query optimizer beyond simple fusion
- Parquet I/O (Arrow IPC only), write-side CSV formatting niceties
- Mutation-in-place API (all ops return new frames; buffers may be shared/COW internally)

## 2. Locked architecture decisions

Record each as `/docs/adr/ADR-00X.md`. The orchestrator may extend but not silently
reverse these; a reversal requires an ADR with benchmark evidence.

**ADR-001 — WASM-core memory ownership.** All column buffers are allocated inside the
wasm module's linear memory via an exported allocator. JS never owns column data; it
holds `TypedArray` views constructed over `memory.buffer`. This makes the JS↔WASM
boundary zero-copy in both directions. Consequence: `memory.grow` detaches all views.
Mitigation: a module-level *generation counter* incremented on every grow; every JS
view accessor checks the counter and lazily rebuilds views. All view access goes
through one `viewOf(column)` helper — no raw view caching anywhere else.

**ADR-002 — Arrow-compatible columnar layout.** Per column: contiguous data buffer +
validity bitmap (1 bit/value, LSB ordering, Arrow convention). Strings are
dictionary-encoded: `i32` index buffer + dictionary (offsets `i32` + UTF-8 bytes),
both in wasm memory. JS-side decoded strings are memoized per dictionary slot, so
each unique string crosses the boundary at most once. Layout compatibility is what
makes Phase 6 Arrow IPC interop nearly free.

**ADR-003 — Hybrid query API.** Expression-first: `col('a').gt(5)` builds a small AST
compiled to a sequence of kernel calls. Lambda escape hatch: `df.filterFn(r => ...)` /
`df.mapFn(...)` iterate rows via a reusable row-proxy over column views (zero-copy,
but scalar JS speed). Docs must clearly label the escape hatch as the slow path and
show the expression equivalent in every example.

**ADR-004 — Dual wasm builds, feature-detected.** Build both SIMD128 and scalar wasm
binaries. At load, probe with `WebAssembly.validate` on a tiny SIMD module and
dynamically import the right binary. Covers Safari < 16.4 and older Node without
shipping SIMD-less performance to everyone.

**ADR-005 — No index; hash-based relational ops.** `groupby` and `join` use a wasm
hash table over column values (64-bit hashes; strings hash their dictionary indices
after dictionary unification). Sort is `argsort` producing an `i32` permutation, then
`gather`.

**ADR-006 — Parallelism is an opt-in shared-memory mode.** Because column data lives
in wasm memory, transferable ArrayBuffers are not available (a `WebAssembly.Memory`
cannot be transferred). v1 main path is single-threaded. A flagged parallel mode
(`enableThreads()`) uses a shared `WebAssembly.Memory` (SAB-backed) + worker pool and
requires cross-origin isolation (COOP/COEP). If isolation is absent, the call warns
and no-ops. Ship in Phase 5; everything before it must not assume threads.

**ADR-007 — Implementation language: decided by Phase 0 spike.** Candidates:
AssemblyScript and Rust. The spike implements the same three kernels (f64 add, f64
null-aware sum, comparison→bitmask filter) in both, SIMD + scalar, and records:
gzipped binary size, throughput at 1M/10M rows, allocator story, string handling
ergonomics, toolchain friction (build time, debugging, CI). Decision criteria in
priority order: (1) kernel throughput, (2) binary size vs budget, (3) maintainability
for agents writing many kernels. Expected outcome for planning purposes: Rust wins
codegen, AssemblyScript wins size/simplicity; either is acceptable behind the ABI.

**ADR-008 — Stable kernel ABI.** All kernels are `export`ed flat C-style functions
operating on pointers + lengths (e.g. `add_f64(lhs_ptr, rhs_ptr, out_ptr, validity_ptr, len)`).
The ABI doc (`/contracts/wasm-abi.md`) is the contract between the API layer and the
kernel layer, and is what allows kernel subagents to work in parallel.

## 3. Data model

**dtypes v1:** `f64`, `f32`, `i32`, `u32`, `bool` (u8 storage; validity bitpacked), `utf8` (dict-encoded).

**Null semantics (pandas-flavored):**
- Arithmetic and comparisons propagate null (null op x → null).
- Boolean `and`/`or` use Kleene three-valued logic.
- Aggregations skip nulls by default (`skipna` semantics); `count` counts non-null; `size` counts rows.
- `fillNull(value)` and `isNull()` provided from v1.

**Casting:** explicit only (`col('a').cast('f32')`). No implicit widening across dtypes
except integer→float in mixed arithmetic (document the exact matrix in `/contracts/dtypes.md`).

## 4. API surface (v1)

Construction / export:
```
DataFrame.fromColumns({a: Float64Array | number[], b: string[]})
DataFrame.fromRecords([{a: 1, b: 'x'}, ...])
DataFrame.fromCSV(text, opts)          // Phase 6
df.toColumns() / df.toRecords() / df.toArrow() / DataFrame.fromArrow(buf)  // Phase 6
```

Core ops:
```
df.select(['a','b'])                    df.drop(['c'])
df.filter(col('a').gt(5).and(col('b').eq('x')))
df.filterFn(r => r.a > 5)               // escape hatch
df.withColumn('c', col('a').mul(2))     // alias: assign
df.sortValues('a', {descending: false})
df.groupby(['b']).agg({a: 'sum', c: ['mean','max']})   // or expr form: {a: col('a').sum()}
df.join(other, {on: 'id', how: 'inner' | 'left'})
df.head(n) / df.tail(n) / df.slice(i, j)
df.col('a')  → Series (zero-copy)       df.shape / df.columns / df.dtypes
df.describe()
```

Expressions: `col`, `lit`; arithmetic `add sub mul div mod neg`; comparison
`gt ge lt le eq ne`; boolean `and or not`; null `isNull fillNull`; `cast`; aggregations
`sum mean min max count nunique std var first last`.

Ergonomics requirements: readable `toString()`/`console.log` table preview; helpful
errors (unknown column names suggest nearest match; dtype mismatches name both dtypes
and the op); full TypeScript types (lightweight generics on column names are a
stretch goal, not a gate).

## 5. Performance targets & benchmark protocol

Built in Phase 0 **before any library code**, so every later merge is measured.

- Harness: `tinybench` in `/bench`, runnable locally and in CI (Node) plus a browser
  runner page. Fixed seeds; datasets at 100K / 1M / 10M rows (numeric) and 1M rows
  with a 10K-unique string column.
- Baselines recorded in P0: (a) naive JS over array-of-objects, (b) hand-written JS
  loops over typed arrays, (c) Arquero, (d) Danfo.js (informational).
- **Per-kernel gate:** wasm SIMD kernel ≥ 1.5× the typed-array JS baseline at 1M rows,
  else the orchestrator investigates before merge (boundary overhead, codegen, layout).
- **End-to-end gate:** `filter → groupby → agg` pipeline at 1M rows ≥ Arquero.
- **Regression gate:** CI fails on > 10% regression vs the stored baseline JSON.
- **Size gate:** CI fails if gzipped JS entry > 25 KB or any wasm binary > 75 KB.

## 6. Repository layout & tooling

```
/contracts/        # *.d.ts interfaces + wasm-abi.md + dtypes.md (orchestrator-owned)
/docs/adr/         # numbered ADRs
/docs/status.md    # orchestrator ledger: task → agent → state → gate results
/bench/            # harness, datasets generators, baseline JSONs
/src/memory/       # allocator bindings, views, generation counter, dict store
/src/kernels/      # JS-side dispatch stubs per kernel family
/src/expr/         # AST, compiler
/src/frame/        # DataFrame, Series, groupby, join API
/src/io/           # csv, json, arrow (P6)
/src/workers/      # parallel mode (P5)
/wasm/             # as/ or rust/ per ADR-007; build scripts for simd+scalar
/tests/            # vitest unit + property tests (fast-check) + conformance fixtures
```

Tooling: TypeScript strict, tsup (dual ESM/CJS + d.ts), vitest, fast-check, tinybench,
GitHub Actions CI (test + bench + size gates), Changesets for versioning.

## 7. Phase plan

Each task is dispatched with the **task brief template** (§8). Agents are
path-restricted; contracts and ADRs are orchestrator-owned and read-only to subagents.

### Phase 0 — Foundation & language spike *(sequential; 1 Opus agent)*

1. Repo scaffold per §6; CI running test/size/bench gates on a hello-world module.
2. Benchmark harness + dataset generators; record all four baselines into `/bench/baselines/`.
3. Language spike per ADR-007: three kernels × {AssemblyScript, Rust} × {SIMD, scalar};
   produce ADR-007 with the numbers and the decision.
4. Write `/contracts/wasm-abi.md` v1 and `/contracts/dtypes.md`.

**Gate:** CI green; baselines committed; ADR-007 merged with data.

### Phase 1 — Memory core *(sequential; 1 Opus agent; do NOT parallelize)*

1. wasm allocator (bump/freelist arena) with alloc/free/realloc exports; leak-check tests.
2. Column representation: data buffer + validity bitmap; creation from JS arrays
   (typed arrays fast path, plain arrays with null detection slow path).
3. Dictionary string store (build, unify two dictionaries, decode-memoization on JS side).
4. View layer: `viewOf()` with generation-counter invalidation; tests that force
   `memory.grow` mid-operation and verify correctness.
5. Schema/dtype registry; zero-copy `slice`.

**Deliverable contract:** `/contracts/memory.d.ts`. **Gate:** property tests
(fast-check round-trips JS ↔ column for every dtype incl. nulls), grow-invalidation
tests, alloc/free cycle leak test, CI green.

### Phase 2 — Kernels *(parallel fan-out; 3–4 Sonnet agents)*

Orchestrator first writes conformance fixtures (input/expected JSON incl. null cases)
and per-kernel bench entries; then dispatches:

- **Agent A — elementwise:** arith, comparison→bitmask, boolean (Kleene), cast, fillNull. Path: `wasm/**/elementwise*`, `src/kernels/elementwise/`.
- **Agent B — reductions:** sum/mean/min/max/count/nunique/std/var/first/last, null-aware, SIMD where profitable. Path: `.../reduce*`.
- **Agent C — selection:** filter-by-bitmask (compaction), gather/take, argsort (stable, multi-key), top-k, slice materialization. Path: `.../select*`.
- **Agent D — relational:** 64-bit hashing, hash groupby (agg via B's kernels), hash join (inner/left), dictionary unification hook. Path: `.../hash*`. *(Depends on B's ABI, not B's implementation.)*

Every kernel ships SIMD + scalar variants and passes its conformance fixtures.

**Gate per agent:** conformance green; per-kernel bench gate (§5); no edits outside
assigned paths. Orchestrator integrates and re-runs the full suite.

### Phase 3 — Expression & API layer *(1–2 agents, sequential-ish; Opus for the compiler)*

1. Expression AST + compiler → kernel-call sequences; simple fusion: `compare → filter`
   emits one mask + one compaction; chained elementwise ops reuse one output buffer.
2. `DataFrame` / `Series` / `GroupBy` classes wiring §4 surface to kernels.
3. Lambda escape hatch: row proxy over `viewOf()` views; string access via memoized decode.
4. Error messages + table pretty-printer.

**Gate:** full API integration tests; end-to-end bench gate vs Arquero (§5); type tests (`tsd` or vitest type-checks).

### Phase 4 — Hardening & performance sweep *(1 Sonnet agent + orchestrator)*

Profile the benchmark matrix; fix cliffs (boundary chatter, dictionary rebuilds,
allocator fragmentation); verify Safari-scalar path; confirm size budgets; expand
fuzz/property tests across the public API.

**Gate:** all §5 gates green on both SIMD and scalar builds; no open P1 bugs.

### Phase 5 — Parallel mode *(1 Sonnet agent; feature-flagged)*

Shared `WebAssembly.Memory` build variant; worker pool; chunked dispatch for
elementwise/reduction kernels; isolation detection with graceful no-op; docs page on
COOP/COEP setup. **Gate:** correctness parity with single-thread; measurable speedup
≥ 1.8× on 4 workers for 10M-row reductions; zero behavior change when flag is off.

### Phase 6 — I/O, docs, release *(parallel; 2 Sonnet agents)*

- **Agent E — I/O:** CSV reader (typed inference, streaming-friendly), JSON records,
  Arrow IPC read/write against `apache-arrow` in tests only (no runtime dep).
- **Agent F — release:** README with honest benchmark table, API docs (typedoc),
  examples (Vite app + Node script), npm packaging checks (`publint`, `arethetypeswrong`),
  wasm asset-loading docs for Vite/webpack/Node, CHANGELOG, `0.1.0` publish dry run.

**Gate:** fresh-clone install→build→test passes; example apps run; publish dry run clean.

## 8. Orchestration protocol (full-auto)

**Orchestrator (Fable) responsibilities:** owns §2 decisions, `/contracts`, and
`/docs/status.md`; writes acceptance tests + conformance fixtures *before* dispatching
any task; dispatches task briefs; reviews diffs; runs integration, bench, and size
gates; merges; keeps ADR log current.

**Task brief template (every dispatch):**
```
Objective:            one sentence
Inputs:               contract files, fixtures, relevant ADRs
Path scope:           globs the agent may modify
Deliverables:         files + exported symbols
Definition of done:   tests listed, bench gate, lint clean, no out-of-scope diffs
Escalation rule:      if a contract is ambiguous or must change, STOP and return a
                      question — do not improvise around it
```

**Subagent rules:** one task at a time; Sonnet by default, Opus for P1 memory core and
the P3 expression compiler; may add tests, never delete or weaken orchestrator-written
acceptance tests; contracts are read-only.

**Failure handling:** gate failure → one automated retry with the failure output
appended to the brief → on second failure, orchestrator triages (fix forward, re-scope,
or split the task). Any deviation from §2 requires a new ADR before code merges.

**Integration cadence:** orchestrator integrates after each subagent completion, and
runs the full gate suite before starting the next phase. Phases 2 and 6 are the only
concurrent fan-outs; all other phases are sequential by design.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `memory.grow` detaches JS views | Generation counter + single `viewOf()` accessor (ADR-001); dedicated tests |
| String boundary cost | Dictionary encoding + per-slot decode memoization; escape-hatch string access documented as slow |
| Safari < 16.4 lacks SIMD | Dual builds + feature detection (ADR-004) |
| wasm memory can't be transferred to workers | Parallelism is SAB-only opt-in (ADR-006); main path single-threaded |
| wasm32 4 GB memory ceiling | Acceptable for browser scope; document; wasm64 is a future ADR |
| Bundler wasm-loading friction | Ship URL-based loading + documented inline-base64 fallback; test in Vite, webpack, Node |
| AssemblyScript codegen underperforms | ADR-007 spike decides with data before any dependent code exists |
| Kernel agents drift from ABI | Orchestrator-owned conformance fixtures run on every merge |

## 10. Open items for the human (Dylan)

1. Package name + npm scope; license (MIT assumed).
2. Confirm Node ≥ 18 floor and evergreen-browser support target.
3. Whether Danfo.js stays in the benchmark table for the public README.
4. v2 wishlist parking lot: dates/timestamps, i64, chunked columns, Parquet, lazy query optimizer, wasm64.
