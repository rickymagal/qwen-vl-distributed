# Distributed Execution Design (Milestone 1)

This document defines the initial distributed execution design for running a **single inference session** of Qwen3-VL across **multiple machines** using **block-wise pipeline parallelism**.

Scope of this document:
- Topology and responsibility boundaries
- Activation + metadata contracts between stages
- KV cache ownership and lifecycle
- Orchestration model (how stages are started and connected)
- What is explicitly out of scope for v1

This is a design/spec document. The implementation is delivered in later milestones.

---

## 1) Execution Model Overview

We split the model into ordered pipeline stages:

- **Stage 0 (Vision + Projector + Embedding)**:
  - Image inputs are assumed to be provided as tensors by the caller
  - Vision encoder forward
  - Multimodal projector
  - Text embedding and initial conditioning tensors

- **Stage 1..N (Transformer Block Ranges)**:
  - Each stage owns a contiguous range of transformer blocks
  - Each stage maintains a **local KV cache** for its block range

- **Final Stage (Last Blocks + LM Head)**:
  - Remaining transformer blocks
  - Output projection (LM head)
  - Logits/top-k sampling (policy depends on runtime requirements)

Stages communicate in order:
`Stage 0 -> Stage 1 -> ... -> Stage N -> Final`

This is pipeline parallelism. The inference session is one logical request flow that traverses all stages.

---

## 2) Stage Responsibilities

### 2.1 Model Partition Boundary

Partition boundaries are defined only at:
- Output of Stage 0 conditioning tensors
- Output of transformer block range boundaries

No partitioning is performed:
- Inside an attention block
- Inside MoE expert routing
- Inside a single transformer block

### 2.2 KV Cache Ownership

Each stage owns KV for its blocks:

- For each generated token step, the stage updates its KV cache
- No stage requests KV from another stage
- KV is not transferred between machines in v1

KV lifecycle:
- Created at request start (or first token)
- Grows by one step per token
- Destroyed when request ends (or evicted by policy in later milestones)

---

## 3) Data Contracts Between Stages

All inter-stage data must be described by two components:

1) **Tensor payload(s)** (activation buffers)
2) **Metadata header** (shape, dtype, layout, offsets, request id, token position)

### 3.1 Activation Tensor Contract (Core)

At minimum, every stage boundary carries:

- `hidden`: activation tensor for the next stage
  - Prefill shape: `[batch, seq, hidden_size]`
  - Decode shape: `[batch, 1, hidden_size]`

Optional additional tensors at Stage 0 boundary:
- `position_ids` / `rope_offsets` (if not derivable locally)

The contract must specify:
- dtype (BF16/FP16 in v1)
- contiguous layout requirement
- row-major order
- byte stride assumptions

### 3.2 Metadata Header (Minimum Fields)

A minimal header (conceptual; implementation may use protobuf/flatbuffers/custom) includes:

- `request_id` (u64)
- `step_kind` enum: PREFILL or DECODE
- `batch` (u32)
- `seq` (u32)
- `hidden_size` (u32)
- `dtype` enum: FP16/BF16
- `layout` enum: CONTIGUOUS
- `token_index` (u32): current decode position
- `stage_from` (u16), `stage_to` (u16)
- `payload_bytes` (u64)
- `crc32` or `xxhash64` (optional in v1; recommended)

---

## 4) Transport and Orchestration

### 4.1 Transport Options (v1 decision: keep simple and debuggable)

We will support one primary transport at first. Candidate choices:

- **TCP sockets with a simple framing protocol**
  - Pros: minimal dependencies, easy to debug, works everywhere
  - Cons: manual reconnect/backpressure

- **gRPC**
  - Pros: standard tooling, streaming, typed messages
  - Cons: more overhead, more build complexity

- **ZeroMQ**
  - Pros: convenient patterns
  - Cons: extra dependency, operational tuning

Milestone 1 outcome:
- The design supports either TCP framing or gRPC streaming.
- The concrete implementation choice is made in Milestone 4 based on integration constraints.

### 4.2 Startup and Connection Model

Orchestrator responsibilities:
- Launch each stage binary on its assigned machine
- Provide each stage with:
  - stage index
  - listen address
  - upstream address (except Stage 0)
  - downstream address (except Final)
  - model partition spec for that stage (block range)

Stages form a simple chain by connecting to downstream and accepting upstream.

---

## 5) Prefill vs Decode

### 5.1 Prefill

Prefill runs a full sequence through the pipeline:
- Stage 0 produces hidden states for the full prompt length
- Each transformer stage processes `[batch, seq, hidden]`
- Each stage populates KV for the full prompt

### 5.2 Decode (Token-by-Token)

Decode iterates token steps:
- Stage 0 embeds the newly produced token (and maintains conditioning state if needed)
- Each stage processes `[batch, 1, hidden]`
- Each stage appends KV for that token step
- Final stage produces logits and selects the next token

---

## 6) Failure Modes and Out-of-Scope Items (v1)

Out of scope for v1:
- Fault tolerance / retries / resharding
- Expert-centric sharding
- Remote KV cache fetch
- Dynamic pipeline scheduling
- Multi-request batching across the pipeline
- Security features (TLS, auth)
- Advanced compression of activations

v1 is designed to be deterministic, correct, and debuggable.

---

## 7) Acceptance Criteria for Milestone 1 (Design Portion)

The distributed design portion of Milestone 1 is complete when:
- Stage boundaries and responsibilities are defined
- Activation and metadata contracts are defined
- KV ownership and lifecycle are specified
- Orchestration model is specified
- Out-of-scope items are explicitly listed

Implementation work is deferred to Milestone 4.
