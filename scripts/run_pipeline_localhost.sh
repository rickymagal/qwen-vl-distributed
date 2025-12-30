#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE0="${STAGE0:-$ROOT/build/stage0}"
STAGE1="${STAGE1:-$ROOT/build/stage1}"
STAGE2="${STAGE2:-$ROOT/build/stage2}"
STAGE3="${STAGE3:-$ROOT/build/stage3}"

if [[ ! -x "$STAGE0" || ! -x "$STAGE1" || ! -x "$STAGE2" || ! -x "$STAGE3" ]]; then
  echo "[pipeline] stage binaries not found or not executable."
  echo "[pipeline] expected:"
  echo "  $STAGE0"
  echo "  $STAGE1"
  echo "  $STAGE2"
  echo "  $STAGE3"
  echo "[pipeline] build first:"
  echo "  mkdir -p build && cd build && cmake .. && cmake --build . -j"
  exit 2
fi

PORT1="${PORT1:-5001}"
PORT2="${PORT2:-5002}"
PORT3="${PORT3:-5003}"

DEVICE0="${DEVICE0:-0}"
DEVICE1="${DEVICE1:-0}"
DEVICE2="${DEVICE2:-0}"
DEVICE3="${DEVICE3:-0}"

STEPS="${STEPS:-1}"

STAGE_COUNT="${STAGE_COUNT:-4}"

# Small smoke config (does not require real weights)
VOCAB="${VOCAB:-4096}"
HIDDEN="${HIDDEN:-256}"
HEADS="${HEADS:-8}"
KV_HEADS="${KV_HEADS:-4}"
MAX_BATCH="${MAX_BATCH:-2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-64}"
DTYPE="${DTYPE:-fp16}"

# Layer ranges per stage (scaffold)
L0_START="${L0_START:-0}"
L0_END="${L0_END:-1}"
L1_START="${L1_START:-1}"
L1_END="${L1_END:-2}"
L2_START="${L2_START:-2}"
L2_END="${L2_END:-3}"
L3_START="${L3_START:-3}"
L3_END="${L3_END:-4}"

pids=()

cleanup() {
  set +e
  for pid in "${pids[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  wait >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[pipeline] starting stage3 (listen ${PORT3})"
"$STAGE3" \
  --stage-id 3 \
  --stage-count "$STAGE_COUNT" \
  --layer-start "$L3_START" \
  --layer-end "$L3_END" \
  --listen-port "$PORT3" \
  --device "$DEVICE3" \
  --steps "$STEPS" \
  --vocab "$VOCAB" \
  --hidden "$HIDDEN" \
  --heads "$HEADS" \
  --kv-heads "$KV_HEADS" \
  --max-batch "$MAX_BATCH" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --dtype "$DTYPE" \
  >"$ROOT/build/stage3.log" 2>&1 &
pids+=("$!")

sleep 0.2

echo "[pipeline] starting stage2 (listen ${PORT2} -> next ${PORT3})"
"$STAGE2" \
  --stage-id 2 \
  --stage-count "$STAGE_COUNT" \
  --layer-start "$L2_START" \
  --layer-end "$L2_END" \
  --listen-port "$PORT2" \
  --next-host 127.0.0.1 \
  --next-port "$PORT3" \
  --device "$DEVICE2" \
  --steps "$STEPS" \
  --vocab "$VOCAB" \
  --hidden "$HIDDEN" \
  --heads "$HEADS" \
  --kv-heads "$KV_HEADS" \
  --max-batch "$MAX_BATCH" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --dtype "$DTYPE" \
  >"$ROOT/build/stage2.log" 2>&1 &
pids+=("$!")

sleep 0.2

echo "[pipeline] starting stage1 (listen ${PORT1} -> next ${PORT2})"
"$STAGE1" \
  --stage-id 1 \
  --stage-count "$STAGE_COUNT" \
  --layer-start "$L1_START" \
  --layer-end "$L1_END" \
  --listen-port "$PORT1" \
  --next-host 127.0.0.1 \
  --next-port "$PORT2" \
  --device "$DEVICE1" \
  --steps "$STEPS" \
  --vocab "$VOCAB" \
  --hidden "$HIDDEN" \
  --heads "$HEADS" \
  --kv-heads "$KV_HEADS" \
  --max-batch "$MAX_BATCH" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --dtype "$DTYPE" \
  >"$ROOT/build/stage1.log" 2>&1 &
pids+=("$!")

sleep 0.4

echo "[pipeline] starting stage0 (connect -> ${PORT1})"
"$STAGE0" \
  --stage-id 0 \
  --stage-count "$STAGE_COUNT" \
  --layer-start "$L0_START" \
  --layer-end "$L0_END" \
  --next-host 127.0.0.1 \
  --next-port "$PORT1" \
  --device "$DEVICE0" \
  --steps "$STEPS" \
  --vocab "$VOCAB" \
  --hidden "$HIDDEN" \
  --heads "$HEADS" \
  --kv-heads "$KV_HEADS" \
  --max-batch "$MAX_BATCH" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --dtype "$DTYPE"

echo "[pipeline] stage0 finished; waiting for others"
wait

echo "[pipeline] done"
echo "[pipeline] logs:"
echo "  build/stage0.log (stdout already shown)"
echo "  build/stage1.log"
echo "  build/stage2.log"
echo "  build/stage3.log"
