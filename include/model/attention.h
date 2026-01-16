// include/model/attention.h
#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>

#include "core/config.h"
#include "core/kv_cache.h"
#include "core/rope.h"
#include "model/rms_norm.h"

namespace qwen {

// Self-attention with optional KV caching and optional RoPE.
// This implementation is correctness-first (no fused kernels) and runs on CUDA.

class AttentionImpl : public torch::nn::Module {
public:
  AttentionImpl(const ModelConfig& cfg, int32_t layer_index_in_stage);

  // x: [B, T, D]
  // attn_mask: optional, either:
  //   - bool keep-mask broadcastable to [B, H, T, S]
  //   - additive float mask broadcastable to [B, H, T, S]
  // cache: optional KV cache owner for this stage
  // pos: current position in sequence for KV append
  // rope: optional precomputed RoPE tables
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

  torch::Tensor& bq() { return wq_->bias; }
  torch::Tensor& bk() { return wk_->bias; }
  torch::Tensor& bv() { return wv_->bias; }
  torch::Tensor& bo() { return wo_->bias; }

  void enable_qk_norm(bool enabled) { use_qk_norm_ = enabled; }
  bool qk_norm_enabled() const { return use_qk_norm_; }
  RmsNorm& q_norm() { return q_norm_; }
  RmsNorm& k_norm() { return k_norm_; }

private:
  ModelConfig cfg_;
  int32_t layer_index_in_stage_ = 0;

  torch::nn::Linear wq_{nullptr};
  torch::nn::Linear wk_{nullptr};
  torch::nn::Linear wv_{nullptr};
  torch::nn::Linear wo_{nullptr};

  RmsNorm q_norm_{nullptr};
  RmsNorm k_norm_{nullptr};
  bool use_qk_norm_ = false;
};

TORCH_MODULE(Attention);

} // namespace qwen
