#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[check] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[check] nvidia-smi"
  nvidia-smi || true
else
  echo "[check] nvidia-smi not found"
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "[check] nvcc --version"
  nvcc --version || true
else
  echo "[check] nvcc not found"
fi

PYTHON_BIN="${PYTHON_BIN:-$ROOT/python_export/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
echo "[check] python torch cuda ($PYTHON_BIN)"
"$PYTHON_BIN" -c "import torch; \
print('torch.__version__:', torch.__version__); \
print('torch.version.cuda:', torch.version.cuda); \
print('torch.cuda.is_available:', torch.cuda.is_available()); \
print('device_count:', torch.cuda.device_count()); \
print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')" || true

BIN="${BIN:-$ROOT/build/distributed_pipeline_stage}"
if [[ -x "$BIN" ]]; then
  echo "[check] ldd $BIN | grep cuda/torch"
  ldd "$BIN" | grep -E "torch|cudart|cuda|nvrtc" || true
else
  echo "[check] binary not found: $BIN"
fi
