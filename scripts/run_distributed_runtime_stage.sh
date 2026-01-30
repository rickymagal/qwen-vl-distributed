#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-$ROOT/build/distributed_pipeline_stage}"
OUT_DIR="${OUT_DIR:-$ROOT/python_export/reduced_export_out}"

HF_CFG="${HF_CFG:-$OUT_DIR/hf_config.json}"
WEIGHTS="${WEIGHTS:-$OUT_DIR/weights.pt}"

NUM_STAGES="${NUM_STAGES:-2}"
STAGE_IDX="${STAGE_IDX:-0}"
LISTEN_PORT="${LISTEN_PORT:-}"
NEXT_HOST="${NEXT_HOST:-}"
NEXT_PORT="${NEXT_PORT:-}"
OUT_FILE="${OUT_FILE:-}"
KV_OUT_FILE="${KV_OUT_FILE:-}"
DEVICE="${DEVICE:-0}"

LAYER_BEGIN="${LAYER_BEGIN:-}"
LAYER_END="${LAYER_END:-}"

SEND_KV="${SEND_KV:-0}"
RECV_KV="${RECV_KV:-0}"
KV_RESTORE="${KV_RESTORE:-0}"

if [[ ! -x "$BIN" ]]; then
  echo "[stage] missing binary: $BIN"
  exit 2
fi

args=(
  --hf-config "$HF_CFG"
  --weights "$WEIGHTS"
  --num-stages "$NUM_STAGES"
  --stage-idx "$STAGE_IDX"
  --device "$DEVICE"
)

if [[ -n "$LISTEN_PORT" ]]; then
  args+=(--listen "$LISTEN_PORT")
fi
if [[ -n "$NEXT_HOST" && -n "$NEXT_PORT" ]]; then
  args+=(--next-host "$NEXT_HOST" --next-port "$NEXT_PORT")
fi
if [[ -n "$OUT_FILE" ]]; then
  args+=(--out "$OUT_FILE")
fi
if [[ -n "$KV_OUT_FILE" ]]; then
  args+=(--kv-out "$KV_OUT_FILE")
fi
if [[ -n "$LAYER_BEGIN" && -n "$LAYER_END" ]]; then
  args+=(--layer-begin "$LAYER_BEGIN" --layer-end "$LAYER_END")
fi
if [[ "$SEND_KV" == "1" ]]; then
  args+=(--send-kv)
fi
if [[ "$RECV_KV" == "1" ]]; then
  args+=(--recv-kv)
fi
if [[ "$KV_RESTORE" == "1" ]]; then
  args+=(--kv-restore)
fi

exec "$BIN" "${args[@]}"
