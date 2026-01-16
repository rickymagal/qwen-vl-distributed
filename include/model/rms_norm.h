#pragma once

#include <torch/torch.h>

#include "core/tensor_utils.h"

namespace qwen {

class RmsNormImpl : public torch::nn::Module {
public:
  explicit RmsNormImpl(int64_t dim, double eps = 1e-6) : eps_(eps) {
    require(dim > 0, "RmsNorm: dim must be > 0");
    weight_ = register_parameter("weight", torch::ones({dim}));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    require(x.defined(), "RmsNorm: x is undefined");
    require(x.dim() >= 2, "RmsNorm: expected x with dim >= 2");
    auto var = x.pow(2).mean(-1, /*keepdim=*/true);
    auto y = x * torch::rsqrt(var + eps_);
    return y * weight_;
  }

  torch::Tensor& weight() { return weight_; }
  const torch::Tensor& weight() const { return weight_; }

private:
  torch::Tensor weight_;
  double eps_;
};

TORCH_MODULE(RmsNorm);

} // namespace qwen
