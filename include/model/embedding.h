#pragma once

#include <torch/torch.h>

#include "core/config.h"

namespace qwen {

class EmbeddingImpl : public torch::nn::Module {
 public:
  explicit EmbeddingImpl(const qwen::ModelConfig& cfg);

  torch::Tensor forward(const torch::Tensor& input_ids);

  const qwen::ModelConfig& cfg() const { return cfg_; }

  // Expose the underlying weight tensor for mapping/debugging.
  torch::Tensor weight() const { return embedding_->weight; }

 private:
  qwen::ModelConfig cfg_;
  torch::nn::Embedding embedding_{nullptr};
};

TORCH_MODULE(Embedding);

}  // namespace qwen
