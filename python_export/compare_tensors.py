#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two torch tensors saved via torch.save")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--atol", type=float, default=5e-3)
    p.add_argument("--rtol", type=float, default=5e-2)
    args = p.parse_args()

    a = torch.load(args.a, map_location="cpu")
    b = torch.load(args.b, map_location="cpu")
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")

    diff = (a - b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (b.abs() + 1e-8)).max().item()

    report = {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "atol": args.atol,
        "rtol": args.rtol,
        "pass": bool((max_abs <= args.atol) or (max_rel <= args.rtol)),
        "shape": list(a.shape),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
