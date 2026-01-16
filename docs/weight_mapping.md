# Weight Mapping (Hugging Face -> C++ / LibTorch)

This document defines how Hugging Face (HF) tensor keys map onto the C++/LibTorch modules in this repository.

It covers:
- What the offline export must produce (`hf_config.json`, `weights.pt`)
- The canonical HF key patterns for Qwen3-style transformer blocks
- The expected C++ module structure and parameter names
- A repeatable loading pattern (including validation and diagnostics)

> Note: This repo intentionally keeps the runtime independent from Python. All key discovery happens offline (export), while runtime uses a thin weight loader that can fetch tensors by HF key.

---

## Export Artifacts

### `hf_config.json`
A frozen subset of the HF config sufficient to build `qwen::ModelConfig`:
- `vocab_size`
- `hidden_size`
- `num_hidden_layers`
- `num_attention_heads`
- MoE fields (if enabled): `num_experts`, `num_experts_per_tok`, etc.
- Vision fields (if enabled): vision encoder dims, projector dims, image token count policy, etc.

### `weights.pt`
A single file containing all tensors addressable by their HF keys.

Recommended packaging for C++:
- A TorchScript container saved via `torch::jit::save` so C++ can load it via `torch::jit::load`.
- Keys must match the original HF `state_dict` keys (no renaming).

---

## C++ Module Structure (Runtime)

At a high level the text decoder follows this structure:

- `Embedding` (token embedding)
- `N x Block` (transformer blocks)
  - Attention (q/k/v/o projections)
  - MLP (dense or MoE)
  - Norms (input + post-attn)
- Final norm
- LM head

The actual C++ classes may have different internal member names, but **the mapping is defined by the HF keys**, not by C++ variable names.

---

## Canonical Key Mapping (Text Decoder)

### Core (outside blocks)

| HF key | Meaning | C++ destination |
|---|---|---|
| `model.language_model.embed_tokens.weight` | token embedding table | `Embedding::weight` |
| `model.language_model.norm.weight` | final RMSNorm weight | final norm parameter |
| `lm_head.weight` | output projection | LM head linear weight |

Some checkpoints use `model.language_model.lm_head.weight` instead of `lm_head.weight`. The loader should try both.

### Per-layer keys (dense MLP)

For each transformer layer index `i` in `[0, num_hidden_layers)`:

| HF key | Meaning |
|---|---|
| `model.language_model.layers.{i}.input_layernorm.weight` | pre-attn RMSNorm |
| `model.language_model.layers.{i}.self_attn.q_proj.weight` | attention Q projection |
| `model.language_model.layers.{i}.self_attn.k_proj.weight` | attention K projection |
| `model.language_model.layers.{i}.self_attn.v_proj.weight` | attention V projection |
| `model.language_model.layers.{i}.self_attn.o_proj.weight` | attention output projection |
| `model.language_model.layers.{i}.self_attn.q_norm.weight` | optional Q RMSNorm (if `use_qk_norm`) |
| `model.language_model.layers.{i}.self_attn.k_norm.weight` | optional K RMSNorm (if `use_qk_norm`) |
| `model.language_model.layers.{i}.post_attention_layernorm.weight` | pre-MLP RMSNorm |
| `model.language_model.layers.{i}.mlp.gate_proj.weight` | MLP gate projection (SwiGLU) |
| `model.language_model.layers.{i}.mlp.up_proj.weight` | MLP up projection |
| `model.language_model.layers.{i}.mlp.down_proj.weight` | MLP down projection |

Biases:
- Many Qwen-family checkpoints are bias-free. If a key is missing (e.g. `...bias`), treat it as expected unless your module declares bias parameters.

### Per-layer keys (MoE MLP)

If MoE is enabled, the MLP portion usually contains:
- A router / gate weight
- Expert weights

Common patterns (the export should preserve the model’s exact keys):

| HF key pattern | Meaning |
|---|---|
| `model.language_model.layers.{i}.mlp.gate.weight` | router weight |
| `model.language_model.layers.{i}.mlp.experts.gate_up_proj` | combined gate+up for all experts (shape varies) |
| `model.language_model.layers.{i}.mlp.experts.down_proj` | expert down projections (shape varies) |

Some variants use `router.weight` instead of `gate.weight`. Support both if your export indicates it.

For Qwen3-VL-235B-A22B, MoE expert weights are grouped:
- `...mlp.experts.gate_up_proj` packs gate+up and must be split before loading.
- `...mlp.experts.down_proj` packs down projections.
The C++ loader accepts either `[E, 2*I, H]` (gate+up) and `[E, H, I]` (down) or transposed equivalents.

---

## Vision Encoder / Projector Mapping

Vision keys vary between Qwen-VL variants and revisions. The mapping rules here are:

1. **Do not guess.** Extract the exact keys from the exported `weights.pt`.
2. Treat the vision encoder and projector as their own namespaces.
3. Keep a single source-of-truth table generated from export.

Suggested approach in export:
- Dump a `weights_manifest.json` containing `{key: {shape, dtype}}` for all tensors.
- The manifest is used to confirm that the C++ side is loading the expected set.

If your HF checkpoint uses a `visual` or `vision_tower` prefix, typical groups include:
- patch embedding / conv stem
- transformer blocks
- final norm
- projector / adapter MLP

Document the exact prefixes in `weights_manifest.json` once the model is available.

---

## Loading Pattern (Recommended)

### 1) Build an explicit mapping table in C++
Define a function that maps “C++ parameter” -> “HF key” and loads it:

```cpp
static void load_param(
    const qwen::PtWeightLoader& wl,
    torch::Tensor param,
    const std::string& hf_key,
    const torch::Device& device) {

  torch::Tensor src = wl.get(hf_key);
  if (!src.defined()) {
    throw std::runtime_error("Missing weight: " + hf_key);
  }

  src = src.to(device, /*dtype=*/param.dtype(), /*non_blocking=*/false).contiguous();
  if (src.sizes() != param.sizes()) {
    throw std::runtime_error("Shape mismatch for " + hf_key);
  }

  param.copy_(src);
}
```

### 2) Apply the table for each module
Example (conceptual):

```cpp
load_param(wl, embedding->weight(), "model.language_model.embed_tokens.weight", device);

for (int i = 0; i < cfg.num_hidden_layers; ++i) {
  load_param(wl, blocks[i]->attn->wq(), "model.language_model.layers."+std::to_string(i)+".self_attn.q_proj.weight", device);
  // ... same pattern for wk, wv, wo, norms, mlp weights ...
}
```

### 3) Validate coverage
At initialization, log:
- total keys available
- keys consumed by mapping
- missing required keys
- unexpected extra keys (warn only)

This is the fastest way to catch naming drift between HF revisions.

---

## Diagnostics Checklist

If forward fails with “tensors on different devices”:
- Ensure `model.to(device)` was called after construction.
- Ensure all loaded weights were copied into CUDA tensors (or moved with the module).
- Ensure inputs (`input_ids`, `hidden`) are allocated on the same CUDA device.

If forward fails with a shape mismatch:
- Confirm `hf_config.json` matches the same `weights.pt` export revision.
- Confirm you are using the correct attention layout (head_dim implied by config).
- For MoE, confirm the number of experts and expert hidden sizes.

---

## Status

- This mapping is **pattern-complete** for the text decoder (Qwen3-style keys).
- Vision mapping becomes fully concrete once the final target checkpoint is exported and `weights_manifest.json` is generated.
