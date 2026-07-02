#!/usr/bin/env bash
# Build script for the Rust WASM spike (ADR-007).
# Run this INSIDE the Docker container:
#   docker run --rm -v <worktree>:/work -w /work dataframe-dev bash -lc \
#     'bash wasm/spike/rust/build.sh'
#
# Outputs (relative to this script's directory):
#   dist/scalar.wasm     — scalar build, wasm-opt -O3
#   dist/simd.wasm       — +simd128 build, wasm-opt -O3
#   dist/scalar.wasm.gz  — gzipped scalar
#   dist/simd.wasm.gz    — gzipped simd

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$SCRIPT_DIR"
DIST_DIR="$CRATE_DIR/dist"
TARGET="wasm32-unknown-unknown"

mkdir -p "$DIST_DIR"

echo "=== Rust WASM Spike Build ==="
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"

# ---- Scalar build ----
echo ""
echo "--- Building scalar ---"
SCALAR_START=$(date +%s%3N)
(
  cd "$CRATE_DIR"
  cargo build --release --target "$TARGET" 2>&1
)
SCALAR_END=$(date +%s%3N)
SCALAR_BUILD_MS=$((SCALAR_END - SCALAR_START))
echo "Scalar build: ${SCALAR_BUILD_MS}ms"

SCALAR_RAW="$CRATE_DIR/target/$TARGET/release/dataframe_spike.wasm"
cp "$SCALAR_RAW" "$DIST_DIR/scalar_raw.wasm"

echo "Running wasm-opt -O3 on scalar..."
wasm-opt -O3 "$DIST_DIR/scalar_raw.wasm" -o "$DIST_DIR/scalar.wasm"
rm "$DIST_DIR/scalar_raw.wasm"

# ---- SIMD build ----
echo ""
echo "--- Building simd (+simd128) ---"
SIMD_START=$(date +%s%3N)
(
  cd "$CRATE_DIR"
  RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target "$TARGET" 2>&1
)
SIMD_END=$(date +%s%3N)
SIMD_BUILD_MS=$((SIMD_END - SIMD_START))
echo "SIMD build: ${SIMD_BUILD_MS}ms"

SIMD_RAW="$CRATE_DIR/target/$TARGET/release/dataframe_spike.wasm"
cp "$SIMD_RAW" "$DIST_DIR/simd_raw.wasm"

echo "Running wasm-opt -O3 on simd..."
wasm-opt -O3 --enable-simd "$DIST_DIR/simd_raw.wasm" -o "$DIST_DIR/simd.wasm" 2>/dev/null || \
  wasm-opt -O3 "$DIST_DIR/simd_raw.wasm" -o "$DIST_DIR/simd.wasm"
rm "$DIST_DIR/simd_raw.wasm"

# ---- Gzip both ----
echo ""
echo "--- Gzip sizes ---"
gzip -k -f "$DIST_DIR/scalar.wasm"
gzip -k -f "$DIST_DIR/simd.wasm"

SCALAR_SIZE=$(wc -c < "$DIST_DIR/scalar.wasm")
SCALAR_GZ=$(wc -c < "$DIST_DIR/scalar.wasm.gz")
SIMD_SIZE=$(wc -c < "$DIST_DIR/simd.wasm")
SIMD_GZ=$(wc -c < "$DIST_DIR/simd.wasm.gz")

echo "scalar.wasm:    ${SCALAR_SIZE} bytes raw, ${SCALAR_GZ} bytes gzipped"
echo "simd.wasm:      ${SIMD_SIZE} bytes raw, ${SIMD_GZ} bytes gzipped"
echo ""

# Write build metadata
cat > "$DIST_DIR/build_meta.json" <<EOF
{
  "scalar_build_ms": $SCALAR_BUILD_MS,
  "simd_build_ms": $SIMD_BUILD_MS,
  "scalar_wasm_bytes": $SCALAR_SIZE,
  "scalar_gz_bytes": $SCALAR_GZ,
  "simd_wasm_bytes": $SIMD_SIZE,
  "simd_gz_bytes": $SIMD_GZ
}
EOF

echo "=== Build complete. Files in $DIST_DIR ==="
ls -lh "$DIST_DIR"
