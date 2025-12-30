#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"

echo "[build_and_smoke] Root: $ROOT_DIR"
echo "[build_and_smoke] Build dir: $BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[build_and_smoke] Configuring with CMake"
cmake .. -DCMAKE_BUILD_TYPE=Release ${TORCH_DIR:+-DTorch_DIR=$TORCH_DIR}

echo "[build_and_smoke] Building"
cmake --build . -j$(nproc)

echo "[build_and_smoke] Running CUDA smoke test"
if [ -f "$ROOT_DIR/scripts/run_smoke_cuda.sh" ]; then
  bash "$ROOT_DIR/scripts/run_smoke_cuda.sh"
else
  echo "ERROR: scripts/run_smoke_cuda.sh not found"
  exit 1
fi

echo "[build_and_smoke] Done"
