#include "model/embedding.h"

#include "core/tensor_utils.h"

namespace qwen {

EmbeddingImpl::EmbeddingImpl(const qwen::ModelConfig& cfg) : cfg_(cfg) {
  qwen::require(cfg_.vocab_size > 0, "Embedding: vocab_size must be > 0");
  qwen::require(cfg_.hidden_size > 0, "Embedding: hidden_size must be > 0");
  embedding_ = register_module(
      "embedding", torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg_.vocab_size, cfg_.hidden_size)));
}

torch::Tensor EmbeddingImpl::forward(const torch::Tensor& input_ids) {
  qwen::require(input_ids.defined(), "Embedding: input_ids is undefined");
  qwen::require_cuda(input_ids, "Embedding: input_ids must be CUDA tensor");
  qwen::require(input_ids.scalar_type() == torch::kInt64, "Embedding: input_ids must be int64");

  return embedding_->forward(input_ids);
}

}  // namespace qwen
