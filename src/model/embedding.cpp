#include "model/embedding.h"
#include "core/tensor_utils.h"

namespace qwen {

EmbeddingImpl::EmbeddingImpl(const ModelConfig& cfg) {
  require(cfg.vocab_size > 0, "Embedding: vocab_size must be > 0");
  require(cfg.hidden_size > 0, "Embedding: hidden_size must be > 0");

  tok_embed_ = register_module(
      "tok_embed",
      torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.vocab_size, cfg.hidden_size)));
}

torch::Tensor EmbeddingImpl::forward(const torch::Tensor& input_ids) {
  require(input_ids.defined(), "Embedding: input_ids is undefined");
  require_cuda(input_ids, "Embedding: input_ids");
  require(input_ids.scalar_type() == torch::kLong, "Embedding: input_ids must be int64");
  require(input_ids.dim() == 2, "Embedding: expected input_ids [B,T]");
  return tok_embed_->forward(input_ids);
}

} // namespace qwen
