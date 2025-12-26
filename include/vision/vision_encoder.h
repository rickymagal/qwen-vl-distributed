#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <string>
#include "core/config.h"

namespace qwen {

// Placeholder vision encoder interface for Milestone 1 scaffolding.
// The real Qwen3-VL vision encoder (and preprocessing) will be locked in spec and
// implemented in src/vision/*.cpp during Milestone 2.
//
// This module is designed to output a sequence of visual tokens/embeddings suitable
// for the multimodal projector and text decoder.

class VisionEncoderImpl : public torch::nn::Module {
public:
  explicit VisionEncoderImpl(const ModelConfig& cfg);

  // images: expected to be CUDA tensor. Shape is model-specific.
  // Returns: visual embeddings, typically [B, V, Dv] on CUDA.
  torch::Tensor forward(const torch::Tensor& images);

  const ModelConfig& cfg() const { return cfg_; }

private:
  ModelConfig cfg_;

  // Minimal placeholder so the module is a valid nn::Module.
  // Replaced by real vision backbone later.
  torch::nn::Linear stub_{nullptr};
};

TORCH_MODULE(VisionEncoder);

} // namespace qwen
