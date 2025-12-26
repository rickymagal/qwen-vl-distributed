# Architecture Overview

This repository contains a CUDA-only LibTorch reimplementation of
Qwen3-VL-235B-A22B-Thinking. The model architecture is manually rewritten
in C++ and does not rely on ONNX, TensorRT, or Python at runtime.

The system is designed to execute a single inference instance across
multiple machines using block-wise pipeline parallelism.

Key characteristics:

- Pure C++ (LibTorch) inference runtime
- Python used only as a one-time build/export step
- Manual implementation of attention, MoE routing, and KV caching
- Explicit ownership of transformer blocks per pipeline stage
- Deterministic execution without dynamic scheduling
