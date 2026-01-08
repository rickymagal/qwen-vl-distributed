# Distributed Execution Design (Pipeline / Block Sharding)

This document describes how a *single* Qwen3-VL inference instance is executed across multiple machines using **pipeline parallelism** (contiguous layer ranges per stage).

The intent is:
- predictable ownership (each stage owns a fixed set of layers and its KV cache)
- minimal cross-machine traffic (only activations + token metadata)
- simple orchestration (no expert-level routing over the network)

---

## Terms

- **Stage**: one process (usually one GPU) that owns a contiguous layer range.
- **Block / Layer**: one transformer layer in the text decoder.
- **Shard**: the layer range `[layer_start, layer_end)` assigned to a stage.
- **KV cache**: per-layer, per-token cached keys/values used for autoregressive decoding.

---

## Stage Topology

The intended topology for Qwen3-VL is:

1. **stage0_vision** (optional): vision encoder + projector, producing initial text-side embeddings
2. **stage0**: early text layers
3. **stage1**: mid text layers
4. **stage2**: mid text layers
5. **stage3**: late text layers
6. **stageN_output**: logits + sampling (may be co-located with the last stage)

Not all deployments need all binaries; the minimal “text-only” pipeline is stages 0..N plus output.

---

## Activation Contract (Between Stages)

Each stage receives a `StageInput` and returns a `StageOutput`:

### Inputs (typical)
- `pos` (int): current token position
- `input_ids` (int64 CUDA tensor): `[batch, seq]` for the first stage
- `hidden` (bf16/fp16 CUDA tensor): `[batch, seq, hidden]` for non-first stages
- generation metadata (optional): attention mask mode, rope cache offsets, etc.

### Outputs (typical)
- `hidden_out` (bf16/fp16 CUDA tensor): `[batch, seq, hidden]`
- KV updates are *local* (owned by the stage) and not transmitted in v1

Rule: **All tensors passed to compute must live on the same CUDA device for that stage.**

---

## Shard Boundary Computation

Shard boundaries can be provided explicitly or computed automatically.

### Explicit
Each stage is launched with:
- `--layer-start <int>`
- `--layer-end <int>`

### Automatic (default)
If not provided, boundaries are computed from:
- `num_hidden_layers` read from `--hf-config /path/hf_config.json`
- `--num-stages <S>`
- `--stage-rank <r>`

Algorithm:
- Split layers as evenly as possible
- Earlier stages receive one extra layer if `L % S != 0`

Pseudo-code:

```cpp
int base = L / S;
int rem  = L % S;

int start = r * base + std::min(r, rem);
int end   = start + base + (r < rem ? 1 : 0);
```

This yields disjoint contiguous ranges covering `[0, L)`.

---

## Concrete Example (Illustrative)

If `num_hidden_layers = 80` and `num_stages = 4`:

| Stage | Layer range |
|---:|---|
| 0 | `[0, 20)` |
| 1 | `[20, 40)` |
| 2 | `[40, 60)` |
| 3 | `[60, 80)` |

If `num_hidden_layers` is not divisible by `num_stages`, the first stages get the remainder layers.

> The *actual* ranges for Qwen3-VL-235B are determined by the exported `hf_config.json` at runtime. The stage binaries print the final computed shard at startup.

---

## Memory Estimates Per Stage

Total GPU memory per stage is approximately:

1. **Weights** for the layers owned by the stage
2. **KV cache** for those layers and the current sequence length
3. Temporary **activations** and workspace

### KV cache sizing (per stage)

Let:
- `B` = batch size (usually 1)
- `T` = current sequence length (prompt + generated so far)
- `H` = hidden size
- `L_s` = number of layers on this stage
- `dtype_bytes` = 2 for fp16/bf16

Typical transformer KV per layer stores K and V of shape `[B, n_heads, T, head_dim]`.
Since `H = n_heads * head_dim`, KV bytes per layer is approximately:

`KV_bytes_per_layer ≈ 2 * B * T * H * dtype_bytes`

So stage KV is:

`KV_bytes_stage ≈ L_s * 2 * B * T * H * dtype_bytes`

This estimate ignores minor padding/metadata.

### Weight sizing (rule of thumb)

If each stage owns ~1/S of the layers, weight memory tends to scale similarly.
For accurate numbers:
- sum `numel * dtype_bytes` over all parameters owned by the stage
- report it at startup (recommended)

---

## Runtime Orchestration

Each stage process does:

1. Parse `--hf-config` and build `ModelConfig`
2. Determine its shard boundaries (explicit or automatic)
3. Load `--weights` and assign tensors to its owned modules
4. Move model to the CUDA device selected by `--device`
5. Enter request loop:
   - receive input (tokens or hidden)
   - run forward for its shard
   - send output hidden to next stage

Transport and serialization are an implementation detail (TCP sockets, gRPC, or custom protocol), but the contract is fixed: **hidden tensors + metadata**.

---

## Failure Modes & Guardrails

- Refuse to start if shard is empty or out of range.
- Print a summary at startup:
  - config (hidden, heads, layers, vocab)
  - shard range
  - device id
  - weight keys loaded (count)
- Validate that all required weights for owned layers exist before serving requests.

---

## Status

- Pipeline sharding model is locked.
- Concrete shard boundaries are computed from `hf_config.json` unless explicitly provided.
- Memory estimates are provided as formulas and should be emitted as runtime logs once full weights are present.
