# Architecture Overview

This repository implements a **CUDA-only inference runtime** for **Qwen3-VL-235B-A22B-Thinking**, rewritten manually in **C++ using LibTorch**. The implementation does **not** rely on ONNX, TensorRT, or Python at runtime. Python is used exclusively as a **one-time export and inspection tool** to extract model metadata, define tensor mappings, and generate reproducible build artifacts.

The system is explicitly designed to execute **a single inference instance distributed across multiple machines**, using **block-wise pipeline parallelism** rather than data parallelism or expert-centric sharding.

---

## High-Level Design Goals

- Deterministic, reproducible inference without dynamic runtime graph construction  
- Explicit ownership of model state per execution stage  
- Minimal runtime dependencies (LibTorch + CUDA only)  
- Clear separation between offline export and online execution  
- Compatibility with multi-process, multi-machine execution  

---

## Model Decomposition

The architecture is decomposed into the following logical components:

### 1. Vision Encoder

The vision encoder processes raw visual inputs into dense embeddings using the original Qwen3-VL visual backbone. All vision layers are executed as a single logical stage, producing fixed-shape embeddings suitable for fusion with text tokens.

Responsibilities:
- Image preprocessing and embedding
- Vision transformer forward pass
- Emission of vision token embeddings

### 2. Multimodal Projector

The multimodal projector maps vision embeddings into the shared embedding space used by the language model. This stage enforces strict shape and dtype compatibility between vision outputs and text token embeddings.

Responsibilities:
- Linear projection into LM embedding space
- Alignment of vision and text token representations

### 3. Transformer Stack (Block-wise)

The language model consists of a large sequence of transformer blocks. Each block is treated as an independently assignable execution unit.

Each transformer block contains:
- Multi-head self-attention with rotary positional embeddings
- Key/value cache logic for autoregressive decoding
- Mixture-of-Experts (MoE) or dense MLP sublayers, depending on block configuration
- Residual connections and normalization layers

Blocks are grouped into **contiguous ranges**, with each range assigned to a pipeline stage.

### 4. Output Head

The final stage applies normalization and the language modeling head to produce logits for token sampling.

---

## Distributed Execution Model

The system uses **pipeline parallelism** to distribute execution across machines:

- Each pipeline stage owns a contiguous range of transformer blocks
- Each stage runs as an independent process (and potentially on a separate machine)
- Stages communicate via explicit activation and KV-cache transfer
- No shared mutable state exists between stages

### Execution Flow

1. Inputs enter the first stage (vision encoder + initial blocks)
2. Intermediate activations are serialized and forwarded to the next stage
3. KV-cache entries are forwarded alongside activations during autoregressive decoding
4. The final stage produces logits and returns tokens to the client

### Determinism and Scheduling

- Execution order is static and predefined
- No dynamic routing, load balancing, or expert reassignment occurs at runtime
- All routing decisions are resolved during export and initialization

---

## Offline vs Runtime Responsibilities

### Offline (Python Export)

- Pin Hugging Face model revision
- Extract architecture configuration and tensor metadata
- Generate tensor-to-shard mapping manifests
- Define block ownership boundaries

### Runtime (C++ / LibTorch)

- Load weights and tensors based on exported manifests
- Execute forward passes deterministically
- Manage KV-cache lifecycle per pipeline stage
- Transfer activations between stages

---

This architecture establishes a stable foundation for subsequent milestones, including full weight loading, numerical parity validation, and distributed runtime execution.
