// src/model/moe.cpp
#include "model/moe.h"

#include "core/tensor_utils.h"

namespace qwen {

MoeImpl::MoeImpl(const ModelConfig& cfg, int32_t layer_index_in_stage)
    : cfg_(cfg), layer_index_in_stage_(layer_index_in_stage) {
  require(cfg_.hidden_size > 0, "Moe: cfg.hidden_size must be set");

  if (cfg_.use_moe) {
    require(cfg_.num_experts > 0, "Moe: cfg.num_experts must be set when use_moe=true");
    require(cfg_.top_k > 0, "Moe: cfg.top_k must be set when use_moe=true");

    router_ = register_module("router", torch::nn::Linear(cfg_.hidden_size, cfg_.num_experts));

    const int64_t h = expert_hidden_dim();
    experts_mods_.reserve((size_t)cfg_.num_experts);
    experts_raw_.reserve((size_t)cfg_.num_experts);

    for (int32_t e = 0; e < cfg_.num_experts; ++e) {
      const std::string name = "expert_" + std::to_string(e);
      auto ex = ExpertMLP(model_dim(), h);
      register_module(name, ex);
      experts_raw_.push_back(ex.ptr().get());
      experts_mods_.push_back(ex);
    }
  } else {
    // Dense MLP fallback (non-MoE).
    const int64_t h = expert_hidden_dim();
    auto dense = ExpertMLP(model_dim(), h);
    register_module("dense_0", dense);
    experts_raw_.push_back(dense.ptr().get());
    experts_mods_.push_back(dense);
  }
}

MoeOutput MoeImpl::forward(const torch::Tensor& x) {
  require(x.defined(), "Moe: x is undefined");
  require_cuda(x, "Moe: x must be CUDA");
  require(x.dim() == 3, "Moe: expected x shape [B, T, D]");
  require(x.size(2) == cfg_.hidden_size, "Moe: hidden_size mismatch");

  MoeOutput out;

  if (!cfg_.use_moe) {
    out.y = experts_mods_[0]->forward(x);
    out.router_logits = torch::Tensor();
    return out;
  }

  // Router logits: [B,T,E]
  auto logits = router_->forward(x);
  out.router_logits = logits;

  // topk over experts along the last dim
  auto topk = torch::topk(logits, cfg_.top_k, /*dim=*/-1, /*largest=*/true, /*sorted=*/false);
  auto topk_vals = std::get<0>(topk); // [B,T,K]
  auto topk_idx  = std::get<1>(topk); // [B,T,K] int64

  // Gating weights per token over the selected experts
  auto gates = torch::softmax(topk_vals, -1); // [B,T,K]

  // Correctness-first dispatch: compute each expert output once, then mask+accumulate.
  // This is not optimized and will be replaced by a fused routing path later.
  const int32_t E = cfg_.num_experts;
  std::vector<torch::Tensor> ex_outs;
  ex_outs.reserve((size_t)E);
  for (int32_t e = 0; e < E; ++e) {
    ex_outs.push_back(experts_mods_[(size_t)e]->forward(x)); // [B,T,D]
  }

  auto y = torch::zeros_like(x);

  for (int32_t k = 0; k < cfg_.top_k; ++k) {
    auto idx_k  = topk_idx.select(-1, k);             // [B,T]
    auto gate_k = gates.select(-1, k).unsqueeze(-1);  // [B,T,1]

    for (int32_t e = 0; e < E; ++e) {
      auto mask = (idx_k == e).unsqueeze(-1); // [B,T,1] bool
      if (!mask.any().item<bool>()) continue;

      auto w = gate_k * mask.to(x.scalar_type()); // [B,T,1]
      y = y + ex_outs[(size_t)e] * w;
    }
  }

  out.y = y;
  return out;
}

} // namespace qwen
