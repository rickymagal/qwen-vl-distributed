# Qwen3-VL Distributed Inference Engine (C++ / LibTorch)

This repository implements a **pure C++ / LibTorch inference runtime** for the Qwen3-VL family, targeting **a single inference instance distributed across multiple machines**. Python is used strictly as a **one-time export/build tool** and is never part of the runtime path.

Development proceeds **milestone by milestone**, with each milestone producing concrete, auditable deliverables. This README is incrementally extended as milestones are completed.

---

## Project Goals

- Manual reimplementation of Qwen3-VL architecture in C++ using LibTorch
- Full vision-language inference support (vision encoder, projector, text decoder)
- Block-wise distributed inference across machines (single logical inference session)
- No ONNX / TensorRT dependency
- Python used only for offline export and inspection
- Reproducible builds and explicit artifact boundaries

---

## Repository Structure (High Level)

```
.
├── CMakeLists.txt
├── cmake/
│   └── TorchConfig.cmake
├── include/
│   ├── core/
│   ├── vision/
│   ├── model/
│   ├── loader/
│   └── runtime/
├── src/
│   ├── core/
│   ├── vision/
│   ├── model/
│   ├── loader/
│   └── runtime/
├── python_export/
│   ├── export_model.py
│   ├── export_config.py
│   └── requirements.txt
├── tests/
├── scripts/
└── docs/
```

---

## Milestone 1 — Architecture Lock, Export Artifacts, and Distributed Execution Design

### Objective

Lock the exact Hugging Face model specification for **Qwen3-VL-235B-A22B-Thinking**, produce stable offline export artifacts consumable by LibTorch C++, and define the initial **distributed execution design** for running one inference session across multiple machines.

### Work Performed

- Pinned the Hugging Face repository and revision for deterministic builds
- Performed model archeology (vision encoder, projector, transformer stack, MoE routing, KV cache behavior, tensor shapes/dtypes)
- Implemented a CPU-only Python export path producing a packed `state_dict` and structured metadata
- Produced an initial distributed execution design for block-wise sharding (pipeline parallelism)

### Key Decisions (Milestone 1)

- Python is **not** part of the runtime and is used only for offline export
- Export artifacts are treated as immutable build inputs
- Runtime is **CUDA-only** and implemented in C++/LibTorch
- Distributed execution uses **block-wise pipeline stages** (not expert-centric)
- Each stage owns its local KV cache for its block range (no remote KV fetch in v1)
- Stage-to-stage transfers use a strict activation contract (dtype/shape/layout) plus metadata

### Deliverables

Export artifacts (generated locally; not checked into git):
- `hf_config.json` — frozen Hugging Face configuration
- `weights.pt` — packed PyTorch state_dict (offline export)
- `weights_manifest.json` — tensor shape/dtype manifest

Distributed design (documented and versioned in repo):
- `docs/distributed_execution_design.md` — stage topology, message contracts, KV ownership, orchestration model

Milestone 1 is considered **complete** once export artifacts are generated successfully and the distributed execution design document is committed.

---

## Milestone 2 — LibTorch Model Rewrite (Planned)

_To be filled after completion._

---

## Milestone 3 — Weight Loading & Parity Validation (Planned)

_To be filled after completion._

---

## Milestone 4 — Distributed Block-Wise Execution Implementation (Planned)

_To be filled after completion._

---

## Milestone 5 — Profiling, Hardening & Handoff (Planned)

_To be filled after completion._
