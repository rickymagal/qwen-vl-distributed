#include "vision/projector.h"
#include "core/tensor_utils.h"

namespace qwen {

ProjectorImpl::ProjectorImpl(const ModelConfig& cfg) : cfg_(cfg) {
  // Milestone 1 placeholder:
  // Project vision hidden dim -> text hidden size.
  const int64_t in_dim = (cfg_.vision_hidden_size > 0) ? cfg_.vision_hidden_size : 1024;
  const int64_t out_dim = (cfg_.hidden_size > 0) ? cfg_.hidden_size : 4096;

  proj_ = register_module("proj", torch::nn::Linear(in_dim, out_dim));
}

torch::Tensor ProjectorImpl::forward(const torch::Tensor& vision_emb) {
  require(vision_emb.defined(), "Projector: vision_emb is undefined");
  require_cuda(vision_emb, "Projector: vision_emb");
  require(vision_emb.dim() == 3, "Projector: expected [B, V, Dv]");

  auto y = proj_->forward(vision_emb);
  return y;
}

} // namespace qwen
