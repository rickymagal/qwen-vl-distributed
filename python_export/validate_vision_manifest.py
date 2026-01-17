#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _load_json(path: Path):
    return json.loads(path.read_text())


def _collect_keys_from_manifest(path: Path) -> set[str]:
    manifest = _load_json(path)
    return set(manifest.keys())


def _collect_keys_from_state_dict(path: Path) -> set[str]:
    import torch

    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        return set(sd.keys())
    raise RuntimeError("weights.pt did not deserialize to a dict")


def _vision_depth(cfg: dict) -> int:
    vcfg = cfg.get("vision_config", {})
    return int(vcfg.get("depth", vcfg.get("num_hidden_layers", 0)) or 0)


def _expected_block_fields() -> Iterable[str]:
    return [
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.proj.weight",
        "attn.proj.bias",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc1.bias",
        "mlp.linear_fc2.weight",
        "mlp.linear_fc2.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Validate vision manifest key coverage")
    p.add_argument("--hf-config", required=True)
    p.add_argument("--manifest", default="")
    p.add_argument("--weights", default="")
    p.add_argument("--min-blocks", type=int, default=1)
    args = p.parse_args()

    cfg = _load_json(Path(args.hf_config))
    keys: set[str]
    if args.manifest:
        keys = _collect_keys_from_manifest(Path(args.manifest))
    elif args.weights:
        keys = _collect_keys_from_state_dict(Path(args.weights))
    else:
        raise SystemExit("Provide --manifest or --weights")

    report = {
        "has_patch_embed": False,
        "has_pos_embed": False,
        "has_merger": False,
        "block_count": 0,
        "missing_fields": [],
    }

    report["has_patch_embed"] = "model.visual.patch_embed.proj.weight" in keys
    report["has_pos_embed"] = "model.visual.pos_embed.weight" in keys
    report["has_merger"] = any(k.startswith("model.visual.merger.") for k in keys) or any(
        k.startswith("model.visual.deepstack_merger_list.") for k in keys
    )

    # Count blocks by index.
    block_indices = set()
    for k in keys:
        if k.startswith("model.visual.blocks."):
            parts = k.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                block_indices.add(int(parts[3]))
    report["block_count"] = len(block_indices)

    # Validate required fields for the first N blocks we expect.
    depth = _vision_depth(cfg)
    expected_blocks = max(args.min_blocks, min(depth if depth > 0 else args.min_blocks, len(block_indices)))
    for i in range(expected_blocks):
        for field in _expected_block_fields():
            key = f"model.visual.blocks.{i}.{field}"
            if key not in keys:
                report["missing_fields"].append(key)

    print(json.dumps(report, indent=2))

    if not report["has_patch_embed"] or not report["has_pos_embed"] or not report["has_merger"]:
        raise SystemExit(2)
    if report["block_count"] < args.min_blocks:
        raise SystemExit(2)
    if report["missing_fields"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
