#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch


def _load_json(path: Path):
    return json.loads(path.read_text())


def _collect_keys_from_manifest(path: Path) -> set[str]:
    manifest = _load_json(path)
    return set(manifest.keys())


def _collect_keys_from_state_dict(path: Path) -> set[str]:
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        return set(sd.keys())
    raise RuntimeError("weights.pt did not deserialize to a dict")


def _text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def _required_text_keys(cfg: dict) -> Iterable[str]:
    tc = _text_cfg(cfg)
    n_layers = int(tc.get("num_hidden_layers", 0))
    use_qk_norm = bool(tc.get("qk_norm", tc.get("use_qk_norm", False)))
    num_experts = int(tc.get("num_experts", 0))
    top_k = int(tc.get("num_experts_per_tok", 0))
    use_moe = num_experts > 0 and top_k > 0

    lm_prefix = "model.language_model"
    yield f"{lm_prefix}.embed_tokens.weight"
    yield f"{lm_prefix}.norm.weight"
    yield "lm_head.weight"
    yield f"{lm_prefix}.lm_head.weight"

    for i in range(n_layers):
        base = f"{lm_prefix}.layers.{i}"
        yield f"{base}.input_layernorm.weight"
        yield f"{base}.post_attention_layernorm.weight"
        yield f"{base}.self_attn.q_proj.weight"
        yield f"{base}.self_attn.k_proj.weight"
        yield f"{base}.self_attn.v_proj.weight"
        yield f"{base}.self_attn.o_proj.weight"
        if use_qk_norm:
            yield f"{base}.self_attn.q_norm.weight"
            yield f"{base}.self_attn.k_norm.weight"
        if use_moe:
            yield f"{base}.mlp.gate.weight"
            yield f"{base}.mlp.experts.gate_up_proj"
            yield f"{base}.mlp.experts.down_proj"
        else:
            yield f"{base}.mlp.gate_proj.weight"
            yield f"{base}.mlp.up_proj.weight"
            yield f"{base}.mlp.down_proj.weight"


def main() -> None:
    p = argparse.ArgumentParser(description="Validate export keys for C++ loader")
    p.add_argument("--hf-config", required=True)
    p.add_argument("--weights", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--allow-missing-vision", action="store_true")
    args = p.parse_args()

    cfg = _load_json(Path(args.hf_config))
    if args.max_layers > 0:
        if "text_config" not in cfg:
            cfg["text_config"] = {}
        cfg["text_config"]["num_hidden_layers"] = args.max_layers
    keys: set[str]
    if args.manifest:
        keys = _collect_keys_from_manifest(Path(args.manifest))
    elif args.weights:
        keys = _collect_keys_from_state_dict(Path(args.weights))
    else:
        raise SystemExit("Provide --manifest or --weights")

    required = set(_required_text_keys(cfg))
    present = required & keys
    missing = sorted(required - keys)

    # lm_head fallback: if one of the two exists, do not count both as missing.
    if "lm_head.weight" in missing and "model.language_model.lm_head.weight" in keys:
        missing.remove("lm_head.weight")
    if "model.language_model.lm_head.weight" in missing and "lm_head.weight" in keys:
        missing.remove("model.language_model.lm_head.weight")

    report = {
        "required_count": len(required),
        "present_count": len(present),
        "missing_count": len(missing),
        "missing_keys": missing[:200],
    }

    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
