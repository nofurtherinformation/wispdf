# Rust WASM Spike Notes (ADR-007)

Measured on Node v22.23.1, Rust 1.96.1, wasm-opt 108, wasm32-unknown-unknown,
Docker (aarch64 host). Date: 2026-07-02.

## Key numbers (1M rows)

| Kernel            | Scalar Mops/s | SIMD Mops/s | Speedup |
|-------------------|--------------|-------------|---------|
| add_f64           | 3 095         | 4 460        | 1.44×   |
| sum_f64_null      | 1 785         | 2 840        | 1.59×   |
| cmp_gt_f64_mask   | 301           | 3 944        | 13.1×   |

Binary sizes (gzipped): scalar=668 B, simd=915 B. Both under the 75 KB budget by ~100×.

## Allocator story

The spike uses a hand-rolled static bump allocator (`alloc` export). Starting address
is hardcoded to 2 MB to land above the 1 MB wasm-ld default stack + data section.
`core::arch::wasm32::memory_grow` grows linear memory on demand. No free/realloc —
good enough for benchmarking (one-time allocations per bench run).

For production: the Phase 1 memory agent will replace this with a proper arena (bump
+ freelist or dlmalloc). Rust's `#[global_allocator]` makes it easy to plug in
`wee_alloc` (< 1 KB overhead) or a custom one — no wasm-bindgen needed.

## SIMD observations

**add_f64 (1.44× speedup):** Modest gain. The scalar build with opt-level=3 already
benefits from wasm-opt autovectorization — wasm-opt converts the scalar loop to
f64x2 instructions. Explicit `f64x2_add` intrinsics in the SIMD build confirm the
same instruction is used, so the ceiling is architecture memory bandwidth.

**sum_f64_null (1.59× speedup):** Conditional SIMD (both-valid fast path, scalar
fallback for nulls). With all-valid data the fast path dominates. The 1.59× gain
matches f64x2 (2 lanes). The horizontal reduction (extracting two lanes and adding)
adds minor overhead.

**cmp_gt_f64_mask (13.1× speedup):** Massive gain. Scalar writes bits one-at-a-time
into output bytes (shift+OR per element), causing heavy dependency chains and poor
throughput (~300 Mops/s). SIMD uses `f64x2_gt` + `i64x2_bitmask` to pack 2 results
per instruction, processing 8 elements per output byte in 4 SIMD ops. The bitmask
output kernel is the strongest ADR-007 argument for SIMD; it will be the hottest path
in `filter()` pipelines.

## String ergonomics

Rust has no native WASM string type. For dictionary-encoded strings (ADR-002):
- The index buffer is `i32` — Rust handles this naturally as a flat `&[i32]`.
- Dictionary decoding happens on the JS side (ADR-002 specifies JS-side memoization).
- From the wasm side, strings are never touched — kernels operate on index integers.

String kernel authoring (e.g., dictionary unification, hash-based groupby) would work
on `i32` index arrays and `u8` byte buffers, both straightforward in Rust `unsafe`.

VS AssemblyScript: AS has built-in `String` with UTF-16 internal encoding and needs a
UTF-8 conversion step. Rust avoids that entirely since it never materializes strings
in wasm — both are roughly equivalent for the dict-encoded pattern.

## Toolchain friction

**Build time:** ~75 ms per build for this tiny crate (incremental). First cold build
(including LLVM codegen) was ~10 s. For CI this is fine; hot rebuilds during kernel
development will be fast.

**Debugging:** `wasm-objdump` and `wasm2wat` work on the output. Rust's `cargo build
--target wasm32-unknown-unknown` produces no `.wat` by default but `wasm2wat` converts
it. Source maps not applicable (no wasm-bindgen). For kernel debugging, `dbg!` macros
don't work in no_std; the pattern is to test kernels through a native `#[test]` build
first, then validate in wasm.

**No `no_std` friction:** The spike is `no_std` to avoid pulling in OS stubs. The only
`no_std` overhead was writing a 3-line `#[panic_handler]`. No `extern crate alloc`
needed since kernels use raw pointers throughout.

**wasm-opt:** The `--enable-simd` flag is required by some wasm-opt versions to avoid
stripping SIMD instructions. Wrapped with a fallback in the build script.

## ADR-007 input summary

- **Throughput:** Rust SIMD kernels deliver 3–5 Gops/s on numeric ops (add_f64,
  cmp_gt) and 13× advantage of explicit SIMD for bitmask output.
- **Binary size:** 668–915 bytes gzipped — trivially within budget.
- **Maintainability for kernel agents:** Raw `unsafe` pointer loops are verbose but
  pattern-consistent. Each kernel is ~20–30 lines. SIMD intrinsics via
  `core::arch::wasm32` are explicit and well-documented.
- **Vs AssemblyScript:** AS has friendlier syntax but less control over SIMD.
  The cmp_gt 13× story is a strong Rust argument for kernels needing bitmask output.
