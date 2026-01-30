#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig


def _save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Create a reduced export with synthetic weights.")
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", default=os.environ.get("HF_REVISION", None))
    p.add_argument("--out-dir", default="reduced_export")
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--config-path", default="", help="Optional local config.json to avoid network")
    p.add_argument("--override-vocab", type=int, default=0)
    p.add_argument("--override-hidden", type=int, default=0)
    p.add_argument("--override-heads", type=int, default=0)
    p.add_argument("--override-kv-heads", type=int, default=0)
    p.add_argument("--override-intermediate", type=int, default=0)
    p.add_argument("--override-moe-intermediate", type=int, default=0)
    p.add_argument("--override-num-experts", type=int, default=0)
    p.add_argument("--override-top-k", type=int, default=0)
    p.add_argument("--override-max-position", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.config_path:
        cfg_dict = json.loads(Path(args.config_path).read_text())
    else:
        cfg = AutoConfig.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
        cfg_dict = cfg.to_dict()

    text_cfg = cfg_dict.get("text_config", {})
    vocab = int(text_cfg.get("vocab_size", cfg_dict.get("vocab_size", 0)))
    hidden = int(text_cfg.get("hidden_size", 0))
    n_layers = int(text_cfg.get("num_hidden_layers", 0))
    n_heads = int(text_cfg.get("num_attention_heads", 0))
    n_kv = int(text_cfg.get("num_key_value_heads", n_heads))
    intermediate = int(text_cfg.get("intermediate_size", 0))
    moe_intermediate = int(text_cfg.get("moe_intermediate_size", intermediate))
    num_experts = int(text_cfg.get("num_experts", 0))
    top_k = int(text_cfg.get("num_experts_per_tok", 0))
    use_qk_norm = bool(text_cfg.get("qk_norm", text_cfg.get("use_qk_norm", False)))
    max_pos = int(text_cfg.get("max_position_embeddings", cfg_dict.get("max_position_embeddings", 0)))

    if vocab <= 0 or hidden <= 0 or n_heads <= 0:
        raise RuntimeError("Config missing required text fields")

    if args.override_vocab > 0:
        vocab = args.override_vocab
    if args.override_hidden > 0:
        hidden = args.override_hidden
    if args.override_heads > 0:
        n_heads = args.override_heads
    if args.override_kv_heads > 0:
        n_kv = args.override_kv_heads
    if args.override_intermediate > 0:
        intermediate = args.override_intermediate
    if args.override_moe_intermediate > 0:
        moe_intermediate = args.override_moe_intermediate
    if args.override_num_experts > 0:
        num_experts = args.override_num_experts
    if args.override_top_k > 0:
        top_k = args.override_top_k
    if args.override_max_position > 0:
        max_pos = args.override_max_position

    n_layers = min(n_layers if n_layers > 0 else args.num_layers, args.num_layers)

    # Update config dict to match overrides so C++ loader sees consistent sizes.
    if "text_config" not in cfg_dict:
        cfg_dict["text_config"] = {}
    cfg_dict["text_config"]["vocab_size"] = vocab
    cfg_dict["text_config"]["hidden_size"] = hidden
    cfg_dict["text_config"]["num_attention_heads"] = n_heads
    cfg_dict["text_config"]["num_key_value_heads"] = n_kv
    cfg_dict["text_config"]["intermediate_size"] = intermediate
    cfg_dict["text_config"]["moe_intermediate_size"] = moe_intermediate
    cfg_dict["text_config"]["num_experts"] = num_experts
    cfg_dict["text_config"]["num_experts_per_tok"] = top_k
    cfg_dict["text_config"]["num_hidden_layers"] = n_layers
    if max_pos > 0:
        cfg_dict["text_config"]["max_position_embeddings"] = max_pos
        cfg_dict["max_position_embeddings"] = max_pos

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "hf_config.json", cfg_dict)
    head_dim = hidden // n_heads
    kv_dim = n_kv * head_dim
    use_moe = num_experts > 0 and top_k > 0

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dt = dtype_map[args.dtype]

    weights = {}
    lm_prefix = "model.language_model"

    weights[f"{lm_prefix}.embed_tokens.weight"] = torch.randn(vocab, hidden, dtype=dt)
    weights[f"{lm_prefix}.norm.weight"] = torch.randn(hidden, dtype=dt)
    weights["lm_head.weight"] = torch.randn(vocab, hidden, dtype=dt)

    for i in range(n_layers):
        base = f"{lm_prefix}.layers.{i}"
        weights[f"{base}.input_layernorm.weight"] = torch.randn(hidden, dtype=dt)
        weights[f"{base}.post_attention_layernorm.weight"] = torch.randn(hidden, dtype=dt)
        weights[f"{base}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden, dtype=dt)
        weights[f"{base}.self_attn.k_proj.weight"] = torch.randn(kv_dim, hidden, dtype=dt)
        weights[f"{base}.self_attn.v_proj.weight"] = torch.randn(kv_dim, hidden, dtype=dt)
        weights[f"{base}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden, dtype=dt)
        if use_qk_norm:
            weights[f"{base}.self_attn.q_norm.weight"] = torch.randn(head_dim, dtype=dt)
            weights[f"{base}.self_attn.k_norm.weight"] = torch.randn(head_dim, dtype=dt)

        if use_moe:
            weights[f"{base}.mlp.gate.weight"] = torch.randn(num_experts, hidden, dtype=dt)
            weights[f"{base}.mlp.experts.gate_up_proj"] = torch.randn(
                num_experts, 2 * moe_intermediate, hidden, dtype=dt
            )
            weights[f"{base}.mlp.experts.down_proj"] = torch.randn(
                num_experts, hidden, moe_intermediate, dtype=dt
            )
        else:
            weights[f"{base}.mlp.gate_proj.weight"] = torch.randn(intermediate, hidden, dtype=dt)
            weights[f"{base}.mlp.up_proj.weight"] = torch.randn(intermediate, hidden, dtype=dt)
            weights[f"{base}.mlp.down_proj.weight"] = torch.randn(hidden, intermediate, dtype=dt)

    torch.save(weights, out_dir / "weights.pt")

    manifest = {}
    for k, v in weights.items():
        manifest[k] = {
            "shape": list(v.shape),
            "dtype": str(v.dtype).replace("torch.", ""),
            "device": str(v.device),
        }
    _save_json(out_dir / "weights_manifest.json", manifest)
    print(f"[reduced_export] wrote {out_dir}")


if __name__ == "__main__":
    main()
