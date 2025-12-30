#include "vision/vision_encoder.h"

#include "core/tensor_utils.h"

namespace qwen {

VisionEncoderImpl::VisionEncoderImpl(const ModelConfig& cfg) : cfg_(cfg) {}

torch::Tensor VisionEncoderImpl::forward(const torch::Tensor& images) {
  require(images.defined(), "VisionEncoder: input is undefined");
  require_cuda(images, "VisionEncoder: input must be CUDA");
  return images;
}

} // namespace qwen
