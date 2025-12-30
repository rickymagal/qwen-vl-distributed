// src/vision/projector.cpp
#include "vision/projector.h"

#include "core/tensor_utils.h"

namespace qwen {

int64_t ProjectorImpl::get_cfg_i64(const ModelConfig&, const char*, int64_t fallback) {
  return fallback;
}

ProjectorImpl::ProjectorImpl(const ModelConfig& cfg) : cfg_(cfg) {
  in_dim_  = (cfg_.vision_hidden_size > 0) ? cfg_.vision_hidden_size : 1024;
  out_dim_ = (cfg_.hidden_size > 0) ? cfg_.hidden_size : 4096;

  // Optional override if later wired in config; for now pick a safe mid size.
  mid_dim_ = get_cfg_i64(cfg_, "projector_hidden_size", out_dim_);

  // Common projector: LN -> Linear -> GELU -> Linear
  norm_ = register_module(
      "norm",
      torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{in_dim_})));

  fc1_ = register_module("fc1", torch::nn::Linear(in_dim_, mid_dim_));
  fc2_ = register_module("fc2", torch::nn::Linear(mid_dim_, out_dim_));

  drop_ = register_module("drop", torch::nn::Dropout(torch::nn::DropoutOptions(0.0)));

  // Stable init for smoke testing.
  {
    torch::NoGradGuard ng;
    const double std = 0.02;

    for (auto& p : this->parameters(/*recurse=*/true)) {
      if (!p.defined()) continue;
      if (p.numel() == 0) continue;
      if (p.dim() >= 2) {
        p.normal_(0.0, std);
      } else {
        p.zero_();
      }
    }
  }
}

torch::Tensor ProjectorImpl::forward(const torch::Tensor& vision_emb) {
  require(vision_emb.defined(), "Projector: vision_emb is undefined");
  require_cuda(vision_emb, "Projector: vision_emb must be CUDA");
  require(vision_emb.dim() == 3, "Projector: expected [B, V, Dv]");
  require(vision_emb.size(2) == in_dim_, "Projector: unexpected Dv (vision hidden size mismatch)");

  auto x = vision_emb;
  // Ensure float-like dtype.
  if (x.scalar_type() != torch::kFloat && x.scalar_type() != torch::kHalf && x.scalar_type() != torch::kBFloat16) {
    x = x.to(torch::kFloat);
  }

  x = norm_->forward(x);
  x = fc1_->forward(x);
  x = torch::gelu(x);
  x = drop_->forward(x);
  x = fc2_->forward(x);

  return x;
}

} // namespace qwen
