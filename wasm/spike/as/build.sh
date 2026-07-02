#!/usr/bin/env bash
# Build script for AssemblyScript spike (run inside Docker)
# Usage (from repo root):
#   docker run --rm -v $(pwd):/work -w /work/wasm/spike/as dataframe-dev bash build.sh
set -euo pipefail

cd "$(dirname "$0")"

BUILD_START_MS=$(date +%s%3N)

echo "=== [AS spike] Installing dependencies ==="
npm install --prefer-offline 2>&1 | grep -v "npm notice"

mkdir -p build

echo ""
echo "=== [AS spike] Building scalar target (asc -O3) ==="
npx asc assembly/scalar.ts \
  --outFile build/scalar-pre.wasm \
  --textFile build/scalar.wat \
  --optimizeLevel 3 \
  --shrinkLevel 0 \
  --noAssert \
  --runtime stub

echo "=== [AS spike] Optimising scalar with wasm-opt -O3 ==="
wasm-opt -O3 build/scalar-pre.wasm -o build/scalar.wasm

echo ""
echo "=== [AS spike] Building SIMD target (asc -O3 --enable simd) ==="
npx asc assembly/simd.ts \
  --outFile build/simd-pre.wasm \
  --textFile build/simd.wat \
  --optimizeLevel 3 \
  --shrinkLevel 0 \
  --noAssert \
  --runtime stub \
  --enable simd

echo "=== [AS spike] Optimising SIMD with wasm-opt -O3 --enable-simd ==="
wasm-opt -O3 --enable-simd build/simd-pre.wasm -o build/simd.wasm

BUILD_END_MS=$(date +%s%3N)
BUILD_DIFF=$((BUILD_END_MS - BUILD_START_MS))
BUILD_TIME_S=$(awk "BEGIN{printf \"%.2f\", $BUILD_DIFF / 1000}")

echo ""
echo "=== Build complete in ${BUILD_TIME_S}s ==="
ls -lh build/scalar.wasm build/simd.wasm

for f in build/scalar.wasm build/simd.wasm; do
  GZ=$(gzip -c "$f" | wc -c)
  echo "$(basename $f) gzip: ${GZ} bytes"
done

echo "BUILD_TIME_S=${BUILD_TIME_S}" > build/build_time.txt
