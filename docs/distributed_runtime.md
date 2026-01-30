# Distributed Runtime (Multi-Machine Pipeline)

This runtime executes **one logical inference** across **multiple machines** using pipeline stages. Each stage is a standalone CUDA binary that receives activations over TCP, runs its local block range, and forwards results. An **optional KV-cache handoff** is supported to enable stateless stages or validation workflows.

## 1) Wire Format

### 1.1 Activation packet

Header (network byte order):
- `int32 version`
- `int32 stage_from`
- `int32 stage_to`
- `uint64 step`
- `uint64 pos`

Payload:
- `hidden` tensor
- `attn_mask` tensor (optional; encoded as an undefined tensor if absent)

### 1.2 KV packet

Header (network byte order):
- `int32 version`
- `int32 stage_from`
- `int32 stage_to`
- `uint64 step`
- `uint64 pos`

Payload:
- `k` tensor (optional)
- `v` tensor (optional)

The packed KV tensors are expected in `[L, B, kv_heads, S, head_dim]` for compatibility with `runtime/kv_wire.{h,cpp}`.

### 1.3 Tensor encoding (shared by activation and KV)

For each tensor:
1. `uint8 defined` (0 = undefined, 1 = defined)
2. If defined:
   - `int32 dtype`
   - `int32 ndim`
   - `uint64 sizes[ndim]`
   - `uint64 nbytes`
   - raw byte payload

On send, CUDA tensors are copied to CPU and made contiguous to ensure a deterministic wire image.

## 2) Runtime Handoff Contract

- One **TCP connection per stage hop per step**.
- The sender transmits **activation first**, then (optionally) KV on the **same connection**.
- The receiver validates metadata and may:
  - Store KV to disk for validation (`--kv-out`), or
  - Restore into its local cache **only when the sender and receiver share the same layer range** (`--kv-restore`).

## 3) Multi-Machine Demo (2 stages)

Prepare a reduced export:
```bash
python3 python_export/reduced_export.py \
  --out python_export/reduced_export_out \
  --override-layers 2 \
  --override-hidden 64 \
  --override-heads 4 \
  --override-kv-heads 2
```

Host B (stage 1 / last stage, listens):
```bash
./build/distributed_pipeline_stage \
  --hf-config python_export/reduced_export_out/hf_config.json \
  --weights python_export/reduced_export_out/weights.pt \
  --num-stages 2 \
  --stage-idx 1 \
  --listen 7001 \
  --out /tmp/pipeline_out.pt \
  --recv-kv \
  --kv-out /tmp/kv_stage0.pt
```

Host A (stage 0 / first stage, sends):
```bash
./build/distributed_pipeline_stage \
  --hf-config python_export/reduced_export_out/hf_config.json \
  --weights python_export/reduced_export_out/weights.pt \
  --num-stages 2 \
  --stage-idx 0 \
  --next-host <HOST_B_IP> \
  --next-port 7001 \
  --send-kv
```

## 4) Test Coverage

- `tests/test_kv_wire.cpp` validates KV pack/restore roundtrip.
- `tests/test_transport_kv.cpp` validates activation + KV TCP transfer determinism.
- `build/distributed_transport_check` provides an end-to-end transport integrity check.

## 5) Helper Scripts

### 5.1 Local two-stage demo

Runs a 2-stage pipeline on one host (localhost TCP), with a reduced export auto-generated if missing.

```bash
scripts/run_distributed_runtime_local.sh
```

If CUDA initialization fails, run the diagnostic helper:
```bash
scripts/check_cuda_env.sh
```

Environment overrides:
- `OUT_DIR` (reduced export output dir)
- `PORT` (listen port)
- `DEVICE0`, `DEVICE1`
- `L0`, `L1` (layer starts for stage0/stage1)
- `OUT_FILE`, `KV_OUT_FILE`
- `FORCE_REDUCED=1` (regenerate reduced export with small max_position)
- `MODEL_ID` (used only if config download is needed)
- `CONFIG_PATH` (local config JSON; defaults to `python_export/minimal_hf_config.json`)

### 5.2 Per-host stage runner

Use on each host in a multi-machine setup (set env vars then run).

Stage 1 (listener):
```bash
NUM_STAGES=2 STAGE_IDX=1 LISTEN_PORT=7001 RECV_KV=1 \
OUT_FILE=/tmp/pipeline_out.pt KV_OUT_FILE=/tmp/kv_stage0.pt \
scripts/run_distributed_runtime_stage.sh
```

Stage 0 (sender):
```bash
NUM_STAGES=2 STAGE_IDX=0 NEXT_HOST=<HOST_B_IP> NEXT_PORT=7001 SEND_KV=1 \
scripts/run_distributed_runtime_stage.sh
```

Optional overrides:
- `HF_CFG`, `WEIGHTS`, `OUT_DIR`
- `LAYER_BEGIN`, `LAYER_END`
- `DEVICE`
- `KV_RESTORE=1` (only if layer ranges are identical on both sides)

## 6) Multi-stage Examples

### 6.1 Three-stage (hosts A, B, C)

Host C (stage 2, last):
```bash
NUM_STAGES=3 STAGE_IDX=2 LISTEN_PORT=7002 RECV_KV=1 \
OUT_FILE=/tmp/pipeline_out.pt KV_OUT_FILE=/tmp/kv_stage1.pt \
scripts/run_distributed_runtime_stage.sh
```

Host B (stage 1, middle):
```bash
NUM_STAGES=3 STAGE_IDX=1 LISTEN_PORT=7001 NEXT_HOST=<HOST_C_IP> NEXT_PORT=7002 \
RECV_KV=1 SEND_KV=1 KV_OUT_FILE=/tmp/kv_stage0.pt \
scripts/run_distributed_runtime_stage.sh
```

Host A (stage 0, first):
```bash
NUM_STAGES=3 STAGE_IDX=0 NEXT_HOST=<HOST_B_IP> NEXT_PORT=7001 SEND_KV=1 \
scripts/run_distributed_runtime_stage.sh
```

### 6.2 Four-stage (hosts A, B, C, D)

Host D (stage 3, last):
```bash
NUM_STAGES=4 STAGE_IDX=3 LISTEN_PORT=7003 RECV_KV=1 \
OUT_FILE=/tmp/pipeline_out.pt KV_OUT_FILE=/tmp/kv_stage2.pt \
scripts/run_distributed_runtime_stage.sh
```

Host C (stage 2):
```bash
NUM_STAGES=4 STAGE_IDX=2 LISTEN_PORT=7002 NEXT_HOST=<HOST_D_IP> NEXT_PORT=7003 \
RECV_KV=1 SEND_KV=1 KV_OUT_FILE=/tmp/kv_stage1.pt \
scripts/run_distributed_runtime_stage.sh
```

Host B (stage 1):
```bash
NUM_STAGES=4 STAGE_IDX=1 LISTEN_PORT=7001 NEXT_HOST=<HOST_C_IP> NEXT_PORT=7002 \
RECV_KV=1 SEND_KV=1 KV_OUT_FILE=/tmp/kv_stage0.pt \
scripts/run_distributed_runtime_stage.sh
```

Host A (stage 0):
```bash
NUM_STAGES=4 STAGE_IDX=0 NEXT_HOST=<HOST_B_IP> NEXT_PORT=7001 SEND_KV=1 \
scripts/run_distributed_runtime_stage.sh
```

## 7) Host Checklist (before running)

- Ensure `build/distributed_pipeline_stage` exists on each host.
- Confirm each host can reach the next host/port (firewall open).
- Copy `python_export/reduced_export_out/` or full export to each host (paths match `HF_CFG`/`WEIGHTS`).
- Set `NUM_STAGES` and `STAGE_IDX` consistently across hosts.
- Start last stage first, then middle stages, then stage 0.
