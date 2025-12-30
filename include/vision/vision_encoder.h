// include/vision/vision_encoder.h
#pragma once

#include <torch/torch.h>
#include <string>
#include "core/config.h"

namespace qwen {

// CUDA-only ViT-style vision encoder used for Milestone 2 structural validation.
// This is a deterministic, fully functional forward graph intended to:
// - Produce a sequence of visual tokens [B, V, Dv] on CUDA
// - Exercise attention/MLP/norm paths in LibTorch
// - Provide stable shapes for the multimodal projector and downstream stages
//
// It is NOT a claim of exact parity with Qwen's vision backbone yet.
// Exact parity and weight mapping are addressed in Milestone 3.

class VisionEncoderImpl : public torch::nn::Module {
public:
  explicit VisionEncoderImpl(const ModelConfig& cfg);

  // images: CUDA float tensor [B, 3, H, W] (H/W may vary; patching uses floor division).
  // Returns: CUDA float tensor [B, V, Dv] where V = 1 + (H/patch)*(W/patch).
  torch::Tensor forward(const torch::Tensor& images);

  const ModelConfig& cfg() const { return cfg_; }

private:
  ModelConfig cfg_;

  // Config-derived (with conservative defaults if cfg lacks fields)
  int64_t patch_size_ = 14;
  int64_t hidden_ = 1024;
  int64_t heads_ = 16;
  int64_t layers_ = 12;
  int64_t mlp_ratio_ = 4;
  double dropout_ = 0.0;

  // Modules / params
  torch::nn::Conv2d patch_embed_{nullptr};    // [B,3,H,W] -> [B,D,H',W']
  torch::Tensor cls_token_;                   // [1,1,D]
  torch::Tensor pos_embed_;                   // [1,1+max_patches,D] (resized/sliced in forward)
  torch::nn::Dropout drop_{nullptr};

  torch::nn::TransformerEncoder encoder_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};

private:
  static int64_t get_cfg_i64(const ModelConfig& cfg, const char* key, int64_t fallback);
  static bool has_attr_i64(const ModelConfig& cfg, const char* key);
};

TORCH_MODULE(VisionEncoder);

} // namespace qwen
