# AssemblyScript Spike Notes – ADR-007 Criteria

## Benchmark Summary (Docker / Node 22, 2025-07)

| kernel | build | 1M ops/s | 10M ops/s | SIMD vs scalar |
|---|---|---|---|---|
| add_f64 | scalar | 2.7B | 2.7B | — |
| add_f64 | simd | 4.4B | 4.4B | **+63%** |
| sum_f64_null | scalar | 4.1B | 4.1B | — |
| sum_f64_null | simd | 3.6B | 3.6B | -12% (see below) |
| cmp_gt_f64_mask | scalar | 5.6B | 5.6B | — |
| cmp_gt_f64_mask | simd | 4.4B | 4.5B | -21% (see below) |

Gzip sizes: scalar=500 B, simd=582 B (both trivially under 75 KB budget).

---

## 1. Allocator Story

A minimal bump allocator is implemented in the module itself (no dependency on GC runtime):

```typescript
let _bump: usize = 0;
function bump_init(): void { if (!_bump) _bump = __heap_base; }
export function alloc(size: usize): usize {
  bump_init();
  const ptr = _bump;
  _bump += (size + 7) & ~(7 as usize);
  const needed_pages = ((_bump + 65535) >> 16) as i32;
  if (needed_pages > memory.size()) memory.grow(needed_pages - memory.size());
  return ptr;
}
```

Key points:
- `__heap_base` is a linker-generated constant that marks the first free byte after
  static data and the stack; reading it avoids clobbering module data.
- Lazy init (`if (!_bump)`) ensures `__heap_base` is read after the data segment is
  placed, which is required for the constant to be valid.
- No `free`. For scratch buffers in kernel calls this is fine; for a real allocator
  we'd use an arena per column operation with bulk-reset between calls.
- `runtime: stub` in asc keeps the binary minimal — no GC overhead. The stub runtime's
  own `__new` is not used; we manage the heap manually for clean ABI control.

---

## 2. String Handling Ergonomics

Not exercised in this spike (kernels are f64-only), but the AS picture:
- Strings are JS `String` objects and cross the boundary as pointers to
  length-prefixed UTF-16 in linear memory. This is **not** Arrow UTF-8.
- For the Arrow dict-encoded `utf8` column layout the module would operate on `i32`
  dictionary indices (fast, SIMD-friendly) and expose raw UTF-8 bytes by pointer+length
  to JS for decode memoization (ADR-002). No AS string type is needed at the kernel level.
- String-heavy code in AS is more awkward than in Rust: AS doesn't have a native
  `str`/`&[u8]` type; you manipulate pointers and byte lengths manually.

---

## 3. SIMD Results Analysis

**add_f64 (+63%):** Clean f64x2.add vectorization works well. Processing two doubles per
iteration halves the inner loop count. This is the ideal SIMD case — two-input,
one-output, no control flow.

**sum_f64_null (-12%):** SIMD accumulator with validity masking (`v128.and`) is slower
than the scalar 4-accumulator unroll. Root cause: the validity-bit extraction for each
pair of elements involves conditional branches over byte boundaries plus two
`i64x2.replace_lane` calls (lane insertion crosses the SIMD/scalar pipeline boundary).
The scalar path's branch predictor handles the all-valid case almost perfectly.

Fix path for production kernels: batch-process 8 elements per validity byte (one byte
of bitmap covers exactly 8 elements). The SIMD version would load the byte once, unpack
8 masks, and process in two v128 groups. Expected outcome: SIMD would win at 1M+ rows.

**cmp_gt_f64_mask (-21%):** The SIMD comparison (`f64x2.gt`) is fast, but extracting
one bit per lane via `i64x2.extract_lane` then OR-ing into a byte is expensive. The
scalar 8-unroll with hard-coded bitmask literals (0x01..0x80) is highly predictable
and cache-friendly, and the branch predictor can speculate on comparisons well. The
`extract_lane` instructions cause scalar-to-SIMD boundary crossings on each iteration.

Fix path: use `i8x16.bitmask` (extracts high bit of each byte lane into an i32 scalar)
or pack comparison results with `i32x4.narrow_i64x2_s`. This would get the bit mask
in 1–2 instructions instead of 8 extract_lane calls.

---

## 4. Toolchain Friction

### Build time
npm install + 2× asc + 2× wasm-opt: approximately 3-5 seconds total.
asc itself is fast (< 1 s per file).

### Error quality
AS type errors are TypeScript-style and precise:
```
ERROR AS200: Conversion from type 'u32' to 'u8' requires an explicit cast.
   if ((bv >> bit) & 1) ...
                ~~~
```
Shift-amount type errors were the main friction point: `usize` variables used as
shift counts require explicit casts to `u32` or `u8`. This affects essentially every
bit-manipulation kernel. In Rust, integer widths are explicit by construction.

### Debugging
- `--textFile` (WAT output) is very readable for verifying SIMD codegen — can confirm
  `f64x2.add` / `v128.and` instructions are present in the SIMD build.
- No native WASM debugger integration; wasm-pack/wasm-bindgen ecosystem is Rust-only.
- AS's `trace()` builtin (imports `wasm:spectest/print_i32`) is available for printf-
  style debugging.

### Separate source files required for SIMD
AS has no preprocessor (`#if`), and v128 intrinsics are syntax errors when compiled
without `--enable simd`. This forces separate source files (or build targets that point
to different entry files). A single source with conditional SIMD paths is not possible.
Compare Rust: a single file with `#[cfg(target_feature = "simd128")]` gates everything.

### No SIMD for free
SIMD requires explicit use of `v128` intrinsics — the compiler does not auto-vectorize.
This is the same as Rust with explicit SIMD; both require manual kernel SIMD writing.
Rust's `std::simd` / `core::arch` offer more abstraction and cross-target portability
than AS's thin v128 wrappers.

### Module size
Both .wasm binaries are under 600 bytes gzipped. AS's minimal output (with `stub`
runtime) is excellent. No garbage collector, no runtime overhead.

---

## 5. Observations for ADR-007 Decision

**Pros:**
- Tiny binary output; AS is the smallest-output language evaluated.
- TypeScript syntax lowers the barrier for JS developers writing kernels.
- Fast compile iteration (< 1 s per file vs Rust's longer incremental builds).
- `stub` runtime gives full control over the allocator without runtime overhead.

**Cons:**
- No auto-vectorization; SIMD requires explicit v128 code AND separate source files.
- Integer type strictness generates many cast errors in bit-manipulation code (8 errors
  on first compile of these kernels). Every shift amount and byte-width operation needs
  explicit casts — more friction than Rust's integer narrowing warnings.
- String handling at kernel level is pointer + length; no Arrow-native type support.
- No standard library for data structures (hash tables, sort) — these must be
  implemented from scratch in AS, whereas Rust has `std` and crates.
- Debugging is harder than Rust (no native debugger, no `dbg!` macro equivalent).
- SIMD benefit materializes only with care about lane-boundary crossings
  (sum_f64_null and cmp_gt_f64_mask showed regression in this spike's implementations).
