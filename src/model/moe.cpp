#include "model/moe.h"
#include "core/tensor_utils.h"

namespace qwen {
namespace {

struct ExpertMLPImpl : public torch::nn::Module {
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};

  ExpertMLPImpl(int64_t in_dim, int64_t hidden_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(in_dim, hidden_dim));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, in_dim));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    auto y = torch::gelu(fc1->forward(x));
    return fc2->forward(y);
  }
};

TORCH_MODULE(ExpertMLP);

} // namespace

MoeImpl::MoeImpl(const ModelConfig& cfg, int32_t layer_index_in_stage)
    : cfg_(cfg), layer_index_in_stage_(layer_index_in_stage) {
  require(cfg_.hidden_size > 0, "Moe: cfg.hidden_size must be set");

  if (cfg_.use_moe) {
    require(cfg_.num_experts > 0, "Moe: cfg.num_experts must be set when use_moe=true");
    require(cfg_.top_k > 0, "Moe: cfg.top_k must be set when use_moe=true");
    router_ = register_module("router", torch::nn::Linear(cfg_.hidden_size, cfg_.num_experts));

    // Placeholder expert MLP sizes (real model may differ; spec lock will set exact dims)
    const int64_t hidden = (cfg_.intermediate_size > 0) ? cfg_.intermediate_size : (cfg_.hidden_size * 4);

    experts_.reserve(cfg_.num_experts);
    for (int32_t e = 0; e < cfg_.num_experts; ++e) {
      auto ex = ExpertMLP(cfg_.hidden_size, hidden);
      register_module("expert_" + std::to_string(e), ex);
      experts_.push_back(ex.get());
    }
  } else {
    // Dense MLP fallback (non-MoE)
    const int64_t hidden = (cfg_.intermediate_size > 0) ? cfg_.intermediate_size : (cfg_.hidden_size * 4);
    auto dense = ExpertMLP(cfg_.hidden_size, hidden);
    register_module("dense", dense);
    experts_.push_back(dense.get());
  }
}

MoeOutput MoeImpl::forward(const torch::Tensor& x) {
  require(x.defined(), "Moe: x is undefined");
  require_cuda(x, "Moe: x");
  require(x.dim() == 3, "Moe: expected x shape [B, T, D]");
  require(x.size(2) == cfg_.hidden_size, "Moe: hidden_size mismatch");

  MoeOutput out;
  if (!cfg_.use_moe) {
    auto* dense = dynamic_cast<ExpertMLPImpl*>(experts_[0]);
    require(dense != nullptr, "Moe: dense expert type mismatch");
    out.y = dense->forward(x);
    out.router_logits = torch::Tensor(); // undefined
    return out;
  }

  // Router logits: [B,T,E]
  auto logits = router_->forward(x);
  out.router_logits = logits;

  // topk over experts
  auto topk = torch::topk(logits, cfg_.top_k, /*dim=*/-1, /*largest=*/true, /*sorted=*/false);
  auto topk_vals = std::get<0>(topk); // [B,T,K]
  auto topk_idx  = std::get<1>(topk); // [B,T,K] int64

  // Softmax over topk values for gating weights
  auto gates = torch::softmax(topk_vals, -1); // [B,T,K]

  // Compute weighted sum of expert outputs.
  // This is correctness-first and not optimized.
  auto y = torch::zeros_like(x);
  for (int32_t k = 0; k < cfg_.top_k; ++k) {
    // idx_k: [B,T]
    auto idx_k = topk_idx.select(-1, k);
    auto gate_k = gates.select(-1, k).unsqueeze(-1); // [B,T,1]

    // For simplicity: loop over experts and apply mask.
    // (Spec lock/runtime will likely replace this with fused dispatch.)
    for (int32_t e = 0; e < cfg_.num_experts; ++e) {
      auto mask = (idx_k == e).unsqueeze(-1); // [B,T,1] bool
      if (!mask.any().item<bool>()) continue;

      auto* ex = dynamic_cast<ExpertMLPImpl*>(experts_[e]);
      require(ex != nullptr, "Moe: expert type mismatch");

      // Apply expert to full x, then mask. (Not fast, but correct scaffold.)
      auto ex_y = ex->forward(x); // [B,T,D]
      y = y + ex_y * (gate_k * mask.to(x.scalar_type()));
    }
  }

  out.y = y;
  return out;
}

} // namespace qwen
