// include/vision/projector.h
#pragma once

#include <torch/torch.h>
#include <string>
#include "core/config.h"

namespace qwen {

// Multimodal projector (vision -> text hidden).
// Milestone 2 implementation provides a real module graph that:
// - Accepts vision embeddings [B, V, Dv]
// - Produces projected embeddings [B, V, Dtext]
// - Runs fully on CUDA
// - Uses a simple, common structure (Linear -> GELU -> Linear) to exercise graph correctness
//
// Exact projector spec and weight mapping are addressed in Milestone 3.

class ProjectorImpl : public torch::nn::Module {
public:
  explicit ProjectorImpl(const ModelConfig& cfg);

  // vision_emb: [B, V, Dv] -> [B, V, Dtext]
  torch::Tensor forward(const torch::Tensor& vision_emb);

  const ModelConfig& cfg() const { return cfg_; }

private:
  ModelConfig cfg_;

  int64_t in_dim_ = 1024;
  int64_t out_dim_ = 4096;
  int64_t mid_dim_ = 4096;

  torch::nn::Linear fc1_{nullptr};
  torch::nn::Linear fc2_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Dropout drop_{nullptr};

private:
  static int64_t get_cfg_i64(const ModelConfig& cfg, const char* key, int64_t fallback);
};

TORCH_MODULE(Projector);

} // namespace qwen
