#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include "core/config.h"
#include "core/kv_cache.h"
#include "core/rope.h"

namespace qwen {

// Minimal attention interface for Milestone 1 scaffolding.
// The implementation will be in src/model/attention.cpp during Milestone 2.
// This declares a standard self-attention module with optional KV caching.

class AttentionImpl : public torch::nn::Module {
public:
  AttentionImpl(const ModelConfig& cfg, int32_t layer_index_in_stage);

  // x: [B, T, D]
  // attn_mask: optional, e.g. [B, 1, T, S] or model-specific
  // cache: optional KV cache owner for this stage
  // pos: current position in sequence for KV append
  torch::Tensor forward(const torch::Tensor& x,
                        const c10::optional<torch::Tensor>& attn_mask,
                        KVCache* cache,
                        int64_t pos,
                        const c10::optional<RopeTables>& rope);

  const ModelConfig& cfg() const { return cfg_; }

  // Expose weights for loader mapping
  torch::Tensor& wq() { return wq_->weight; }
  torch::Tensor& wk() { return wk_->weight; }
  torch::Tensor& wv() { return wv_->weight; }
  torch::Tensor& wo() { return wo_->weight; }

private:
  ModelConfig cfg_;
  int32_t layer_index_in_stage_ = 0;

  // Linear projections (placeholder shapes; finalized in spec lock)
  torch::nn::Linear wq_{nullptr};
  torch::nn::Linear wk_{nullptr};
  torch::nn::Linear wv_{nullptr};
  torch::nn::Linear wo_{nullptr};
};

TORCH_MODULE(Attention);

} // namespace qwen
