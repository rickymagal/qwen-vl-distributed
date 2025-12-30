// include/model/moe.h
#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <vector>
#include <string>

#include "core/config.h"

namespace qwen {

// MoE / MLP block interface.
// For Qwen3-VL-235B-A22B this is MoE-enabled; we define a correctness-first
// implementation that exercises routing + expert execution on CUDA.
// Weight mapping is handled in Milestone 3.

struct MoeOutput {
  torch::Tensor y;               // [B, T, D]
  torch::Tensor router_logits;   // [B, T, E] (defined only when use_moe=true)
};

struct ExpertMLPImpl : public torch::nn::Module {
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};

  ExpertMLPImpl(int64_t model_dim, int64_t hidden_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(model_dim, hidden_dim));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, model_dim));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    auto y = fc1->forward(x);
    y = torch::gelu(y);
    y = fc2->forward(y);
    return y;
  }
};

TORCH_MODULE(ExpertMLP);

class MoeImpl : public torch::nn::Module {
public:
  MoeImpl(const ModelConfig& cfg, int32_t layer_index_in_stage);

  // x: [B, T, D]
  MoeOutput forward(const torch::Tensor& x);

  const ModelConfig& cfg() const { return cfg_; }

  // Router weights access (for loader mapping later)
  torch::Tensor& router_w() { return router_->weight; }
  torch::Tensor& router_b() { return router_->bias; }

  // Expose raw expert module pointers for loader mapping.
  // (Order is stable: expert_0..expert_{E-1}, or dense_0 for non-MoE.)
  std::vector<torch::nn::Module*>& experts() { return experts_raw_; }

private:
  ModelConfig cfg_;
  int32_t layer_index_in_stage_ = 0;

  // Router: D -> num_experts (only when use_moe=true)
  torch::nn::Linear router_{nullptr};

  // Owning expert modules (stable registration names)
  std::vector<ExpertMLP> experts_mods_;
  std::vector<torch::nn::Module*> experts_raw_;

private:
  int64_t model_dim() const { return cfg_.hidden_size; }
  int64_t expert_hidden_dim() const {
    if (cfg_.intermediate_size > 0) return cfg_.intermediate_size;
    return cfg_.hidden_size * 4;
  }
};

TORCH_MODULE(Moe);

} // namespace qwen
