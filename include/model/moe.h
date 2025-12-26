#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <vector>
#include "core/config.h"

namespace qwen {

// MoE / MLP block interface.
// For Qwen3-VL-235B-A22B this is MoE-enabled; we declare the contract here.
// Implementation will live in src/model/moe.cpp during Milestone 2.

struct MoeOutput {
  torch::Tensor y;                // [B, T, D]
  torch::Tensor router_logits;    // optional debug/analysis
};

class MoeImpl : public torch::nn::Module {
public:
  MoeImpl(const ModelConfig& cfg, int32_t layer_index_in_stage);

  // x: [B, T, D]
  MoeOutput forward(const torch::Tensor& x);

  const ModelConfig& cfg() const { return cfg_; }

  // Router weights access
  torch::Tensor& router_w() { return router_->weight; }
  torch::Tensor& router_b() { return router_->bias; }

  // Expert weights are model-dependent; keep accessor list for loader mapping later.
  // We expose a vector of expert MLP modules for assignment by key mapping.
  std::vector<torch::nn::Module*>& experts() { return experts_; }

private:
  ModelConfig cfg_;
  int32_t layer_index_in_stage_ = 0;

  // Router: D -> num_experts
  torch::nn::Linear router_{nullptr};

  // Placeholder expert modules. The real expert MLP layers will be defined in implementation.
  std::vector<torch::nn::Module*> experts_;
};

TORCH_MODULE(Moe);

} // namespace qwen
