#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <cstdint>

namespace qwen {

// Minimal RoPE helper for later integration.
// This header provides:
//  - precompute_cos_sin: builds cos/sin tables on CUDA
//  - apply_rope: applies RoPE to q/k using cos/sin
//
// Assumptions for apply_rope (common layout):
//  q, k: [B, H, T, D] where D is head_dim
//  cos, sin: [T, D_rope] where D_rope <= D and usually even
//
// This is correctness-first and not kernel-fused.

struct RopeTables {
  torch::Tensor cos; // [T, D_rope]
  torch::Tensor sin; // [T, D_rope]
  int64_t rope_dim = 0;
};

RopeTables precompute_cos_sin(int64_t seq_len,
                              int64_t rope_dim,
                              double theta,
                              c10::ScalarType dtype,
                              int device_index);

void apply_rope_inplace(torch::Tensor q,
                        torch::Tensor k,
                        const RopeTables& tables,
                        int64_t start_pos = 0);

} // namespace qwen
