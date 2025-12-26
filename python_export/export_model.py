#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoConfig, AutoModel


@dataclass
class ExportCfg:
    model_id: str
    revision: Optional[str]
    out_dir: Path
    dtype: str
    trust_remote_code: bool


def _parse_dtype(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {s}")


def _save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def export_model(cfg: ExportCfg) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Milestone-1 rule: do not use device_map / tp_plan / torch.set_default_device.
    # This keeps the export step independent of `accelerate`.
    # We load on CPU and export a packed state_dict and config metadata.
    dtype = _parse_dtype(cfg.dtype)

    print(f"[export] model_id={cfg.model_id}")
    print(f"[export] revision={cfg.revision or 'default'}")
    print(f"[export] out_dir={cfg.out_dir}")
    print(f"[export] dtype={dtype}")

    config = AutoConfig.from_pretrained(
        cfg.model_id,
        revision=cfg.revision,
        trust_remote_code=cfg.trust_remote_code,
    )

    # Save raw HF config as JSON (stringly-typed, stable for C++ parsing)
    # AutoConfig is not always JSON-serializable directly; use to_dict().
    config_dict: Dict[str, Any] = config.to_dict()
    _save_json(cfg.out_dir / "hf_config.json", config_dict)

    # Load model on CPU only to avoid accelerate/device_map requirements.
    # The runtime CUDA requirement is satisfied by LibTorch in C++, not by this export.
    with torch.no_grad():
        model = AutoModel.from_pretrained(
            cfg.model_id,
            revision=cfg.revision,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        model.eval()
        model.to("cpu")

        # Packed state_dict export: torch.save(dict[str, Tensor], path)
        sd = model.state_dict()
        out_sd = cfg.out_dir / "weights.pt"
        print(f"[export] writing packed state_dict -> {out_sd}")
        torch.save(sd, out_sd)

        # Optional: emit a simple key->(shape,dtype) manifest for sanity checks
        manifest: Dict[str, Any] = {}
        for k, v in sd.items():
            try:
                manifest[k] = {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype).replace("torch.", ""),
                    "device": str(v.device),
                }
            except Exception:
                manifest[k] = {"error": "uninspectable"}
        _save_json(cfg.out_dir / "weights_manifest.json", manifest)

    print("[export] done")


def main() -> None:
    p = argparse.ArgumentParser(description="Export HF model artifacts for LibTorch C++ runtime.")
    p.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model id, e.g. Qwen/Qwen3-VL-235B-A22B-Thinking",
    )
    p.add_argument(
        "--revision",
        default=os.environ.get("HF_REVISION", None),
        help="Optional HF revision/commit/tag to pin.",
    )
    p.add_argument(
        "--out-dir",
        default=os.environ.get("EXPORT_OUT_DIR", "export_out"),
        help="Output directory for artifacts.",
    )
    p.add_argument(
        "--dtype",
        default=os.environ.get("EXPORT_DTYPE", "fp16"),
        choices=["fp32", "fp16", "bf16"],
        help="dtype to request when loading weights (export happens on CPU).",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Allow loading custom model code from the repo.",
    )
    args = p.parse_args()

    cfg = ExportCfg(
        model_id=args.model_id,
        revision=args.revision,
        out_dir=Path(args.out_dir).resolve(),
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    export_model(cfg)


if __name__ == "__main__":
    main()
