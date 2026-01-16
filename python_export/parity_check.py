#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel


def _save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Parity check: Python HF vs C++ parity_runner")
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", default=os.environ.get("HF_REVISION", None))
    p.add_argument("--hf-config", required=True, help="Path to exported hf_config.json")
    p.add_argument("--weights", required=True, help="Path to exported weights.pt")
    p.add_argument("--cpp-bin", required=True, help="Path to build/parity_runner")
    p.add_argument("--out-dir", default="parity_out")
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--atol", type=float, default=5e-3)
    p.add_argument("--rtol", type=float, default=5e-2)
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    config = AutoConfig.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.eval()
    model.to(args.device)

    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None and hasattr(config, "text_config"):
        vocab_size = getattr(config.text_config, "vocab_size", None)
    if not vocab_size:
        raise RuntimeError("Unable to resolve vocab_size from config")

    input_ids = torch.randint(0, vocab_size, (1, args.seq_len), device=args.device, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids)
        ref = getattr(out, "logits", None)
        if ref is None:
            ref = getattr(out, "last_hidden_state", None)
        if ref is None:
            raise RuntimeError("Model output has no logits or last_hidden_state")

    input_ids_path = out_dir / "input_ids.pt"
    ref_path = out_dir / "ref.pt"
    cpp_out_path = out_dir / "cpp_out.pt"

    torch.save(input_ids.cpu(), input_ids_path)
    torch.save(ref.cpu(), ref_path)

    cmd = [
        args.cpp_bin,
        "--hf-config",
        args.hf_config,
        "--weights",
        args.weights,
        "--out",
        str(cpp_out_path),
        "--input-ids",
        str(input_ids_path),
        "--device",
        args.device.replace("cuda:", ""),
    ]
    subprocess.run(cmd, check=True)

    cpp_out = torch.load(cpp_out_path)
    ref = ref.cpu()

    diff = (cpp_out - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (ref.abs() + 1e-8)).max().item()

    report = {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "atol": args.atol,
        "rtol": args.rtol,
        "pass": bool((max_abs <= args.atol) or (max_rel <= args.rtol)),
        "ref_shape": list(ref.shape),
        "cpp_shape": list(cpp_out.shape),
    }
    _save_json(out_dir / "parity_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
