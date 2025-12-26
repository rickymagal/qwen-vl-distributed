#pragma once

#include <torch/torch.h>
#include "core/config.h"

namespace qwen {

class EmbeddingImpl : public torch::nn::Module {
public:
  explicit EmbeddingImpl(const ModelConfig& cfg);

  // input_ids: [B, T] int64
  // returns: [B, T, D]
  torch::Tensor forward(const torch::Tensor& input_ids);

  torch::Tensor& weight() { return tok_embed_->weight; }
  const ModelConfig& cfg() const { return cfg_; }

private:
  ModelConfig cfg_;
  torch::nn::Embedding tok_embed_{nullptr};
};

TORCH_MODULE(Embedding);

} // namespace qwen
