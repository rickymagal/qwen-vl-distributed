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

## Milestone 2 — LibTorch Model Rewrite (Completed)

### Objective

Implement a working C++/LibTorch codebase that mirrors the Qwen3-VL high-level module boundaries (vision encoder → projector → text stack), plus the runtime scaffolding required for pipeline-stage execution, so we can begin weight-loading and parity work as soon as the exported artifacts are available.

### Work Performed

- Implemented the core LibTorch module skeletons:
  - Vision encoder module (torch::nn-based) and projector wiring
  - Text-side components (embedding, transformer block wrapper, stage model container)
- Implemented runtime stage scaffolding:
  - `PipelineStage` execution path for “run local” vs “run from activation”
  - Activation packet types and stage I/O structs used to carry tensors + metadata between stages
- Added stage executables under `stages/` and ensured they build end-to-end:
  - `stage0_vision`, `stage0`, `stage1`, `stage2`, `stage3`, `stage1_blocks`, `stage2_blocks`, `stageN_output`
- Added a CUDA smoke forward binary (`tests/test_smoke_forward.cu`) that exercises the LibTorch module wiring on GPU and provides early device/layout validation.
- Fixed incremental build issues found during compilation (namespace and header/impl mismatches) so the full tree builds with the LibTorch toolchain discovered via `torch.utils.cmake_prefix_path`.

### Notes / Current Limitations (by design)

- Real inference is blocked until Milestone 3 because the exported model artifacts are not yet present on disk (weights are too large; SSD pending).
- `stage0` intentionally refuses to run without initialized embedding/weights; this is expected until the weight loader is wired in Milestone 3.
- The CUDA smoke test is an execution sanity-check (device/layout + wiring), not a parity/correctness test yet.

### Deliverables

Buildable C++ runtime + stage binaries:
- `libqwen_core.a`
- `stage0_vision`, `stage0`, `stage1`, `stage2`, `stage3`
- `stage1_blocks`, `stage2_blocks`, `stageN_output`

Smoke test:
- `tests/test_smoke_forward.cu` (built as `test_smoke_forward_cuda`)

Milestone 2 is considered **complete** once the full project builds successfully and the stage executables + CUDA smoke test are produced from a clean build directory.

---

## Milestone 3 — Weight Loading & Parity Validation (Planned)

_To be filled after completion._

---

## Milestone 4 — Distributed Block-Wise Execution Implementation (Planned)

_To be filled after completion._

---

## Milestone 5 — Profiling, Hardening & Handoff (Planned)

_To be filled after completion._
