#include "vision/vision_encoder.h"
#include "core/tensor_utils.h"

namespace qwen {

VisionEncoderImpl::VisionEncoderImpl(const ModelConfig& cfg) : cfg_(cfg) {
  // Milestone 1 placeholder:
  // Map a single pooled image vector to a token embedding.
  //
  // Real encoder will be implemented after spec lock.
  const int64_t out_dim = (cfg_.vision_hidden_size > 0) ? cfg_.vision_hidden_size : 1024;
  stub_ = register_module("stub", torch::nn::Linear(/*in=*/3, /*out=*/out_dim));
}

torch::Tensor VisionEncoderImpl::forward(const torch::Tensor& images) {
  require(images.defined(), "VisionEncoder: images is undefined");
  require_cuda(images, "VisionEncoder: images");
  require(images.dim() == 4, "VisionEncoder: expected images shape [B, C, H, W]");
  require(images.size(1) >= 3, "VisionEncoder: expected C >= 3");

  // Very simple pooling: mean over H,W, take first 3 channels -> [B, 3]
  auto x = images.index({torch::indexing::Slice(),
                         torch::indexing::Slice(0, 3),
                         torch::indexing::Slice(),
                         torch::indexing::Slice()});
  x = x.mean({2, 3}); // [B,3]

  auto y = stub_->forward(x); // [B, out_dim]

  // Produce a single visual token: [B, 1, out_dim]
  return y.unsqueeze(1);
}

} // namespace qwen
