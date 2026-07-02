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

echo ""
echo "--- sizes ---"
for f in scalar simd; do
  raw=$(wc -c < "$OUT_DIR/${f}.wasm")
  gz=$(gzip -c "$OUT_DIR/${f}.wasm" | wc -c)
  printf '%-12s %8d bytes raw   %7d bytes gzipped\n' "${f}.wasm" "$raw" "$gz"
done

echo "=== done -> $OUT_DIR ==="
