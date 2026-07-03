# ADR-011 — Parquet I/O via a scoped subpath export

**Status:** accepted
**Date:** 2026-07-03
**Deciding agent:** v2 LEAD (bead dataframe-dh9.1), commissioned by Dylan for 0.2.0.
**Reverses:** the v1 non-goal "Parquet I/O (Arrow IPC only)" (spec §0 non-goals).
**Depends on:** ADR-009 (i64) + ADR-010 (temporals) for the dtype mapping.

---

## Context

v1 shipped Arrow IPC read/write and explicitly excluded Parquet ("Parquet I/O (Arrow
IPC only)", spec §0; parking lot: "Large binary format dependency"). The blocker was
never capability — it was the **size budget**: the main entry must stay ≤ 25 KB gzipped
and dependency-free (design goal §1). A full Parquet codec (compression, encodings,
Thrift metadata) cannot fit that budget and cannot be a dependency of the main entry.

We reverse the non-goal now because Parquet is the dominant on-disk interchange format
for the data-lake / analytics tooling v2 targets (DuckDB, Polars, Spark, pandas), and
because a **subpath export** lets us add it **without touching the main-entry budget or
its zero-dependency guarantee**.

## Decision

Parquet support ships **only** as a subpath export, **`databonk/parquet`**, with runtime
dependencies permitted **solely** in that subpath.

### Packaging & dependency scope

- **Main entry (`.`) stays dependency-free and ≤ 25 KB gzipped.** It gains **no**
  Parquet code and **no** new dependencies. The size gate on `dist/index.js`
  (`status.md` gate table) is unchanged. This mirrors the existing `databonk/workers`
  subpath precedent (a subpath entry keeps the main bundle small).
- **`databonk/parquet`** is a new `exports` map entry (import/require + `types`, and a
  `typesVersions` alias for node10, matching the `workers` subpath). Only importing
  `databonk/parquet` pulls the Parquet dependencies into a bundle.
- **Runtime dependencies (subpath only):**
  - **`hyparquet`** — the reader. Ships a **native Snappy** decompressor
    (`hyparquet/src/snappy.js`), so Snappy read needs **no** extra codec dependency.
  - **`hyparquet-writer`** — the writer. Also ships native Snappy and defaults to the
    `UNCOMPRESSED` codec.
  - These become `dependencies` (or `optionalDependencies` + a helpful "install
    hyparquet" error) of the package, but are reachable **only** through the subpath.
- **`parquet-wasm` is a `devDependency` only** — a round-trip **test oracle**
  (dh9.2/dh9.7 conformance), never shipped. Its unpacked size (~20 MB, mostly a wasm
  blob) is exactly why it is not a runtime dep.
- **`hyparquet-compressors` is NOT a dependency** (see sizes below): it pulls
  gzip/brotli/zstd/lz4 and would more than triple the subpath. Those codecs are out of
  the supported profile.

### Measured sizes (bundled + minified via esbuild, gzip -9; Docker `dataframe-dev`)

| bundle | raw | **gzipped** |
|---|---:|---:|
| `hyparquet` reader only | 45,927 B | **≈ 13.9 KB** (14,276 B) |
| `hyparquet-writer` only | 56,372 B | **≈ 16.2 KB** (16,616 B) |
| **reader + writer (the shipped subpath profile)** | 100,544 B | **≈ 28.9 KB** (29,555 B) |
| reader + writer + `hyparquet-compressors` (rejected) | 215,444 B | ≈ 101.7 KB (104,133 B) |

The shipped `databonk/parquet` subpath is therefore **≈ 29 KB gzipped** of runtime
dependency, entirely off the main-entry budget. Adding the full codec set would push it
past ~100 KB — the concrete reason exotic codecs are excluded rather than bundled.

### Supported profile (and the failure mode outside it)

`databonk/parquet` supports, for both read and write:

- **dtypes** (via the ADR-009/010 mapping):

  | databonk dtype | Parquet physical / logical |
  |---|---|
  | `f64` | `DOUBLE` |
  | `f32` | `FLOAT` |
  | `i32` | `INT32` |
  | `u32` | `INT32` + `INTEGER(bitWidth=32, signed=false)` logical |
  | `i64` | `INT64` |
  | `bool` | `BOOLEAN` |
  | `utf8` | `BYTE_ARRAY` + `STRING(UTF8)`, dictionary-encoded |
  | `date32` | `INT32` + `DATE` logical |
  | `timestamp` | `INT64` + `TIMESTAMP(MILLIS, isAdjustedToUTC)`; the tz string round-trips in the logical-type/`key_value_metadata` |

- **nulls** via Parquet definition levels ↔ our validity bitmap.
- **dictionary-encoded `utf8`** (maps directly onto our dict-encoded columns, ADR-002).
- **compression: Snappy and uncompressed only** (both native to hyparquet).

**Anything outside the profile raises a clear, specific "unsupported" error** naming
what was unsupported — never a silent wrong result. Explicitly out of profile:
gzip/brotli/zstd/lz4/lzo compression; `INT96` timestamps; nested/repeated (`LIST`/`MAP`/
`STRUCT`) columns; `FIXED_LEN_BYTE_ARRAY`; `DECIMAL`; encodings other than
plain/dictionary/RLE that hyparquet cannot decode.

### API shape (detail owned by dh9.7; ADR fixes the contract)

- `readParquet(bytes: Uint8Array | ArrayBuffer): Promise<DataFrame>` — **async**
  (hyparquet's read path is Promise-based). Builds databonk columns directly from
  hyparquet's column output (avoid materializing row objects) so the import is
  column-native and zero-copy-friendly into wasm memory.
- `writeParquet(df: DataFrame, opts?: { compression?: 'snappy' | 'uncompressed' }):
  Uint8Array` — default `uncompressed`; `snappy` uses the writer's native codec.
- Unsupported dtype/codec/structure → a thrown `Error` from the subpath (not the main
  entry).

## Consequences

- Zero change to the main-entry size gate or its zero-dependency guarantee; the entire
  Parquet cost lives behind an explicit `import 'databonk/parquet'`.
- Tree-shaking / bundlers: apps that never import the subpath ship none of the ~29 KB.
- The supported-profile boundary is a **first-class error surface**, tested against the
  `parquet-wasm` oracle for the in-profile cases and asserted to throw for the
  out-of-profile ones (dh9.2/dh9.7 fixtures).
- Version pinning: `hyparquet` and `hyparquet-writer` are pinned in `package.json`; the
  measured sizes above are the baseline a future dependency bump is checked against.
- A publish/packaging check (publint/attw, as in P6.F) must confirm the new subpath's
  `exports`/`types`/`typesVersions` resolve under node10/node16/bundler.

## Alternatives rejected

- **Parquet in the main entry:** blows the ≤ 25 KB dep-free budget outright. The whole
  reason for a subpath. Rejected.
- **`parquet-wasm` as the runtime engine:** ~20 MB unpacked, a large wasm blob, a
  second wasm module to load alongside our own — wrong size class for a "lightweight"
  library. Kept as a **devDependency test oracle** only. Rejected as runtime.
- **Hand-rolled Parquet codec:** Thrift metadata + multiple encodings + Snappy is a
  large, bug-prone surface duplicating a maintained library, and would itself blow the
  subpath budget once complete. Rejected.
- **Bundle `hyparquet-compressors` for full codec coverage:** +~72 KB gzipped
  (29 → ~102 KB) for gzip/brotli/zstd/lz4, which most analytics Parquet does not need
  (Snappy dominates). Excluded; exotic codecs raise the unsupported error instead.
  Rejected as default.
- **Arrow-IPC-only (status quo):** Arrow IPC is not what the data-lake ecosystem writes
  to disk; users need Parquet interchange. The subpath makes it cheap enough to add.
  Rejected.
