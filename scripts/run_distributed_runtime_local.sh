#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-$ROOT/build/distributed_pipeline_stage}"
OUT_DIR="${OUT_DIR:-$ROOT/python_export/reduced_export_out}"
OUT_FILE="${OUT_FILE:-/tmp/pipeline_out.pt}"
KV_OUT_FILE="${KV_OUT_FILE:-/tmp/kv_stage0.pt}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/python_export/.venv/bin/python}"

PORT="${PORT:-7001}"
DEVICE0="${DEVICE0:-0}"
DEVICE1="${DEVICE1:-0}"

L0="${L0:-0}"
L1="${L1:-1}"

if [[ ! -x "$BIN" ]]; then
  echo "[runtime] missing binary: $BIN"
  echo "[runtime] build first: cmake --build build -j\"$(nproc)\""
  exit 2
fi

if [[ ! -f "$OUT_DIR/hf_config.json" || ! -f "$OUT_DIR/weights.pt" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[runtime] python not found: $PYTHON_BIN"
    exit 2
  fi
  echo "[runtime] reduced export not found, generating -> $OUT_DIR"
  "$PYTHON_BIN" python_export/reduced_export.py \
    --out "$OUT_DIR" \
    --override-layers 2 \
    --override-hidden 64 \
    --override-heads 4 \
    --override-kv-heads 2
fi

pids=()
cleanup() {
  set +e
  for pid in "${pids[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  wait >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[runtime] stage1 listen on :${PORT}"
"$BIN" \
  --hf-config "$OUT_DIR/hf_config.json" \
  --weights "$OUT_DIR/weights.pt" \
  --num-stages 2 \
  --stage-idx 1 \
  --listen "$PORT" \
  --out "$OUT_FILE" \
  --recv-kv \
  --kv-out "$KV_OUT_FILE" \
  --device "$DEVICE1" \
  --layer-begin "$L1" \
  --layer-end "$((L1 + 1))" \
  >"$ROOT/build/distributed_stage1.log" 2>&1 &
pids+=("$!")

sleep 0.2

echo "[runtime] stage0 send -> localhost:${PORT}"
"$BIN" \
  --hf-config "$OUT_DIR/hf_config.json" \
  --weights "$OUT_DIR/weights.pt" \
  --num-stages 2 \
  --stage-idx 0 \
  --next-host 127.0.0.1 \
  --next-port "$PORT" \
  --send-kv \
  --device "$DEVICE0" \
  --layer-begin "$L0" \
  --layer-end "$((L0 + 1))"

echo "[runtime] stage0 finished; waiting for stage1"
wait

echo "[runtime] done"
echo "[runtime] output: $OUT_FILE"
echo "[runtime] kv: $KV_OUT_FILE"
