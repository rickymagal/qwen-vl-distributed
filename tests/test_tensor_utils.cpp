#include "mini_test.h"

#include <torch/torch.h>

int main() {
  // Basic sanity checks around tensor construction and reshape.
  // This test uses LibTorch directly.
  torch::Tensor a = torch::ones({2, 3}, torch::TensorOptions().dtype(torch::kFloat32));

  CHECK_EQ((int64_t)a.dim(), (int64_t)2);
  CHECK_EQ((int64_t)a.size(0), (int64_t)2);
  CHECK_EQ((int64_t)a.size(1), (int64_t)3);
  CHECK_EQ((int64_t)a.numel(), (int64_t)6);
  CHECK_EQ((int64_t)a.scalar_type(), (int64_t)torch::kFloat32);

  torch::Tensor b = a.reshape({3, 2});

  CHECK_EQ((int64_t)b.dim(), (int64_t)2);
  CHECK_EQ((int64_t)b.size(0), (int64_t)3);
  CHECK_EQ((int64_t)b.size(1), (int64_t)2);
  CHECK_EQ((int64_t)b.numel(), (int64_t)6);

  float sum = b.sum().item<float>();
  CHECK_EQ((int64_t)(sum + 0.5f), (int64_t)6);

  return 0;
}
