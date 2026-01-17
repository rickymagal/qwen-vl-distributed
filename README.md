# Qwen-VL Distributed (LibTorch/CUDA)

This repo contains a C++/LibTorch implementation scaffold for running **Qwen3-VL-235B-A22B-Thinking** in a **pipeline-parallel, multi-process / multi-machine** setting. The goal of Phase 1+2 is to lock architecture and export artifacts, then validate a CUDA-only forward pass that can execute stage-by-stage with explicit shard boundaries.

## Status

### Milestone 1 — Architecture spec lock, export artifact, and distributed execution design (DONE)

Delivered:
- **Frozen architecture + control-flow spec** for the multimodal stack (vision encoder → projector → transformer/MoE). See `docs/architecture.md`.
- **Distributed execution design** with ownership rules and KV/activation transport semantics. See `docs/distributed_execution_design.md`.
- **Python → C++ tensor-key mapping** describing how HuggingFace keys map to C++ modules/parameters. See `docs/weight_mapping.md`.
- **Export artifact contract**: the C++ side is defined to consume:
  - `hf_config.json` (model hyperparameters/config)
  - `weights.pt` (packed Torch state_dict)
  - (optional) `tensor_map.json` / index metadata, if provided by the exporter

Phase 1 resubmission gaps called out in review are addressed as follows:

1) **Config loading integration**
- Implemented JSON → `HfConfig` parsing and conversion into the runtime `ModelConfig`.
- Entry points:
  - `include/core/hf_config.h`, `src/core/hf_config.cpp`
  - `qwen::load_hf_config_json(path)` and `qwen::model_config_from_hf_config(hf)`

Example pattern (used by the stage binaries):
- Load `hf_config.json`
- Convert to `ModelConfig`
- Derive shard config via `config_for_stage(...)`

2) **Weight mapping documentation / code**
- The full mapping pattern is documented in `docs/weight_mapping.md`.
- The stage binaries demonstrate the intended usage:
  - `stages/stage0/main.cpp` (vision + embeddings + early blocks)
  - `stages/stage1/main.cpp`, `stages/stage2/main.cpp`, `stages/stage3/main.cpp`
- The loader infrastructure is in `src/loader/*` (`PtWeightLoader`, TorchScript loader, etc.).

3) **Concrete shard boundaries + memory estimates**
- `docs/distributed_execution_design.md` now includes a concrete table for **S=2/4/8** giving:
  - layer ranges per stage (derived from the sharding plan)
  - order-of-magnitude per-stage memory (weights + KV formula with explicit assumptions)

### Milestone 2 — LibTorch CUDA model with distributed pipeline boundaries (DONE)

Delivered:
- **CUDA-only LibTorch forward pass structure** for the multimodal stack:
  - transformer blocks, attention, MoE routing, residuals/norms
  - KV cache logic and stage-by-stage execution hooks
- **Explicit pipeline stages** with reproducible stage binaries:
  - `stage0_vision`, `stage0`, `stage1`, `stage2`, `stage3`, `stageN_output`
  - block-focused runners: `stage1_blocks`, `stage2_blocks`
- **Reproducible CMake build** and passing unit tests.

### Milestone 3 — Weight loading and numerical parity validation (DONE)

Delivered:
- **C++ weight loader with strict checks** (shape/dtype validation, missing/mismatch reporting) and MoE gate_up/down handling.
- **Export validation + parity tooling**:
  - `python_export/validate_export.py` (key coverage checks)
  - `python_export/validate_vision_manifest.py` (vision key coverage without full weights)
  - `python_export/parity_check.py` + `build/parity_runner` (parity harness)
  - `docs/parity_validation.md` (workflow and reduced-export report)
- **Reduced export validation report** recorded in `docs/parity_validation.md`.

## Repository layout

- `include/` — public headers (`core/`, `model/`, `runtime/`, `vision/`, `loader/`)
- `src/` — implementation
- `stages/` — stage executables used to validate pipeline boundaries
- `tests/` — unit tests (CUDA/CPU as applicable)
- `docs/` — architecture, mapping, and distributed design documents

## Build (CMake + LibTorch)

This project uses the **LibTorch shipped inside your Python torch install**.

Run this first to get the correct CMake prefix for your environment:

```bash
python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
```

Then build:

```bash
cd /path/to/qwen-vl-distributed

# Optional but recommended: keep your local torch in a venv (matches your run environment).
# source python_export/.venv/bin/activate

rm -rf build
CMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

cmake --build build -j"$(nproc)"
```

Notes:
- CUDA is autodetected; the build will add the appropriate `-gencode` for your GPU.
- Warnings about NVTX/kineto can appear depending on your torch build; they are not required for these milestones.

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

## Run stage binaries (pipeline boundary validation)

The stage binaries are intentionally simple runners that validate:
- config loading (`hf_config.json` → `ModelConfig`)
- sharding plan and per-stage config derivation
- stage-local forward execution on CUDA (placeholder weights where applicable)

Typical usage (flags may vary slightly per stage binary; see `--help`):

```bash
./build/stage0_vision --help
./build/stage0 --help
./build/stage1 --help
./build/stage2 --help
./build/stage3 --help
./build/stageN_output --help
```

## Docs (what to read first)

- `docs/architecture.md` — architecture/spec lock
- `docs/weight_mapping.md` — HuggingFace tensor-key → C++ parameter mapping
- `docs/distributed_execution_design.md` — distributed plan + concrete shard tables for S=2/4/8
- `docs/parity_validation.md` — parity workflow and reporting
