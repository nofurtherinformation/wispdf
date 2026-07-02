#!/usr/bin/env bash
# Build the production dataframe-wasm crate into scalar.wasm + simd.wasm.
#
# Run INSIDE the Docker dev image:
#   docker run --rm -v <worktree>:/work -w /work dataframe-dev \
#     bash -lc 'bash wasm/rust/build.sh'
#
# Outputs (both feature-detected + loaded by src/memory/loader.ts, and copied
# into dist/ by scripts/copy-wasm.mjs so the size gate covers the real binaries):
#   wasm/dist/scalar.wasm   — scalar build, wasm-opt -O3
#   wasm/dist/simd.wasm     — +simd128 build, wasm-opt -O3 --enable-simd
#
# Both binaries share one source (ABI §1): SIMD paths are gated by
# #[cfg(target_feature = "simd128")]; scalar/SIMD must be behaviourally identical.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$SCRIPT_DIR"
OUT_DIR="$CRATE_DIR/../dist"           # wasm/dist
TARGET="wasm32-unknown-unknown"
RAW="$CRATE_DIR/target/$TARGET/release/dataframe_wasm.wasm"

mkdir -p "$OUT_DIR"

echo "=== dataframe-wasm build ==="
echo "rustc:    $(rustc --version)"
echo "wasm-opt: $(wasm-opt --version)"

# Rust 1.96's wasm32-unknown-unknown target emits the modern post-MVP baseline
# (bulk-memory for memory.copy, sign-ext, nontrapping-fptoint, mutable-globals,
# multivalue, reference-types). wasm-opt must be told to accept them. All are in
# our supported floor (Node >= 18, evergreen browsers, Safari >= 15).
FEATURES="--enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals --enable-multivalue --enable-reference-types"

build_variant() {
  local name="$1"; shift          # scalar | simd
  local rustflags="$1"; shift      # RUSTFLAGS value
  local optflags="$1"; shift       # extra wasm-opt flags

  echo ""
  echo "--- building ${name} ---"
  ( cd "$CRATE_DIR" && RUSTFLAGS="$rustflags" cargo build --release --target "$TARGET" )

  echo "wasm-opt -O3 ${FEATURES} ${optflags} -> ${name}.wasm"
  # shellcheck disable=SC2086
  wasm-opt -O3 $FEATURES $optflags "$RAW" -o "$OUT_DIR/${name}.wasm"
}

build_variant scalar "" ""
build_variant simd   "-C target-feature=+simd128" "--enable-simd"

# ---------------------------------------------------------------------------
# simd-threads variant (ADR-006 / P5.1): SAB-backed shared WebAssembly.Memory.
#
# Requires nightly Rust + rust-src component (to rebuild core with +atomics):
#   rustup toolchain install nightly
#   rustup target add wasm32-unknown-unknown --toolchain nightly
#   rustup component add rust-src --toolchain nightly
#
# Linker flags:
#   --import-memory   : memory imported from host (JS creates WebAssembly.Memory {shared:true})
#   --shared-memory   : declares the imported memory as shared (SAB-backed)
#   --max-memory=N    : required for shared memories (N = 1 GiB = 16384 pages × 65536 B)
#
# All our static mut variables are zero-initialized (BSS); wasm linear memory is
# zero-initialized by default, so there are no active data segments and multiple
# instantiations with the same shared memory are safe — workers never corrupt the
# main thread's arena state even without passive-segments handling.
#
# Workers NEVER call alloc/free/realloc; only the main thread uses the arena.
# This is the invariant that makes shared-memory parallel dispatch safe (ADR-006).
# ---------------------------------------------------------------------------

THREADS_RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
  -C link-arg=--import-memory \
  -C link-arg=--shared-memory \
  -C link-arg=--max-memory=1073741824"

# Install nightly toolchain + rust-src if not already present.
# (The container starts with stable only; these installs are idempotent.)
if ! rustup toolchain list | grep -q '^nightly'; then
  echo "--- installing nightly toolchain ---"
  rustup toolchain install nightly
fi
if ! rustup target list --toolchain nightly --installed | grep -q 'wasm32-unknown-unknown'; then
  echo "--- adding wasm32 target for nightly ---"
  rustup target add wasm32-unknown-unknown --toolchain nightly
fi
if ! rustup component list --toolchain nightly --installed | grep -q 'rust-src'; then
  echo "--- adding rust-src for nightly ---"
  rustup component add rust-src --toolchain nightly
fi

echo ""
echo "--- building simd-threads ---"
( cd "$CRATE_DIR" && RUSTFLAGS="$THREADS_RUSTFLAGS" \
    cargo +nightly build --release --target "$TARGET" \
    -Z build-std=core )

# wasm-opt: enable threads (atomics + bulk-memory) + simd in addition to base features.
# --enable-threads covers memory.atomic.* and i32.atomic.* instructions used by
# the passive-segment init code that wasm-ld injects when --shared-memory is set.
echo "wasm-opt -O3 ${FEATURES} --enable-simd --enable-threads -> simd-threads.wasm"
# shellcheck disable=SC2086
wasm-opt -O3 $FEATURES --enable-simd --enable-threads "$RAW" -o "$OUT_DIR/simd-threads.wasm"

echo ""
echo "--- sizes ---"
for f in scalar simd simd-threads; do
  raw=$(wc -c < "$OUT_DIR/${f}.wasm")
  gz=$(gzip -c "$OUT_DIR/${f}.wasm" | wc -c)
  printf '%-20s %8d bytes raw   %7d bytes gzipped\n' "${f}.wasm" "$raw" "$gz"
done

echo "=== done -> $OUT_DIR ==="
