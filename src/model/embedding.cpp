#include "model/embedding.h"
#include "core/tensor_utils.h"

namespace qwen {

EmbeddingImpl::EmbeddingImpl(const ModelConfig& cfg) : cfg_(cfg) {
  require(cfg_.vocab_size > 0, "Embedding: cfg.vocab_size must be set");
  require(cfg_.hidden_size > 0, "Embedding: cfg.hidden_size must be set");

  tok_embed_ = register_module(
      "tok_embed",
      torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg_.vocab_size, cfg_.hidden_size)));
}

torch::Tensor EmbeddingImpl::forward(const torch::Tensor& input_ids) {
  require(input_ids.defined(), "Embedding: input_ids is undefined");
  require(input_ids.dim() == 2, "Embedding: expected input_ids shape [B, T]");
  require(input_ids.scalar_type() == torch::kInt64, "Embedding: input_ids must be int64");
  require(input_ids.is_cuda(), "Embedding: input_ids must be CUDA tensor");

  return tok_embed_->forward(input_ids);
}

} // namespace qwen
a
