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
