#include "vision/vision_encoder.h"

#include "core/tensor_utils.h"

namespace qwen {

int64_t VisionEncoderImpl::get_cfg_i64(const ModelConfig&, const char*, int64_t fallback) {
  return fallback;
}

bool VisionEncoderImpl::has_attr_i64(const ModelConfig&, const char*) {
  return false;
}

VisionEncoderImpl::VisionEncoderImpl(const ModelConfig& cfg) : cfg_(cfg) {
  if (cfg_.vision_hidden_size > 0) hidden_ = cfg_.vision_hidden_size;
  if (cfg_.vision_num_layers > 0) layers_ = cfg_.vision_num_layers;
  if (cfg_.vision_num_heads > 0) heads_ = cfg_.vision_num_heads;
  if (cfg_.vision_patch_size > 0) patch_size_ = cfg_.vision_patch_size;
  if (cfg_.vision_intermediate_size > 0 && cfg_.vision_hidden_size > 0) {
    const int64_t ratio = cfg_.vision_intermediate_size / cfg_.vision_hidden_size;
    if (ratio > 0) mlp_ratio_ = ratio;
  }

  patch_embed_ = register_module(
      "patch_embed",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(3, hidden_, patch_size_).stride(patch_size_).bias(false)));

  cls_token_ = register_parameter("cls_token", torch::zeros({1, 1, hidden_}));

  // Default to a 224x224 grid if no config hints are present.
  const int64_t max_grid = (224 / patch_size_);
  const int64_t max_patches = max_grid * max_grid;
  pos_embed_ = register_parameter("pos_embed", torch::zeros({1, 1 + max_patches, hidden_}));

  drop_ = register_module("drop", torch::nn::Dropout(torch::nn::DropoutOptions(dropout_)));

  auto enc_opts = torch::nn::TransformerEncoderLayerOptions(hidden_, heads_)
                      .dim_feedforward(hidden_ * mlp_ratio_)
                      .dropout(dropout_);
  auto enc_layer = torch::nn::TransformerEncoderLayer(enc_opts);
  encoder_ = register_module("encoder", torch::nn::TransformerEncoder(enc_layer, layers_));

  norm_ = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_})));
}

torch::Tensor VisionEncoderImpl::forward(const torch::Tensor& images) {
  require(images.defined(), "VisionEncoder: input is undefined");
  require_cuda(images, "VisionEncoder: input must be CUDA");
  require(images.dim() == 4, "VisionEncoder: expected [B, 3, H, W]");

  auto x = images;
  if (x.scalar_type() != torch::kFloat && x.scalar_type() != torch::kHalf && x.scalar_type() != torch::kBFloat16) {
    x = x.to(torch::kFloat);
  }

  x = patch_embed_->forward(x); // [B, D, H', W']
  x = x.flatten(2).transpose(1, 2).contiguous(); // [B, N, D]

  const int64_t B = x.size(0);
  const int64_t N = x.size(1);

  auto cls = cls_token_.to(x.device()).to(x.scalar_type()).expand({B, 1, hidden_});
  x = torch::cat({cls, x}, 1); // [B, 1+N, D]

  torch::Tensor pos;
  if (pos_embed_.defined() && pos_embed_.size(1) >= x.size(1)) {
    pos = pos_embed_.index({torch::indexing::Slice(),
                            torch::indexing::Slice(0, x.size(1)),
                            torch::indexing::Slice()});
  } else {
    pos = torch::zeros({1, 1 + N, hidden_}, x.options());
  }
  pos = pos.to(x.device()).to(x.scalar_type());
  x = x + pos;

  x = drop_->forward(x);
  x = x.transpose(0, 1).contiguous(); // [N, B, D]
  x = encoder_->forward(x);
  x = x.transpose(0, 1).contiguous(); // [B, N, D]
  x = norm_->forward(x);
  return x;
}

} // namespace qwen
