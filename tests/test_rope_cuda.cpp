#include "mini_test.h"

#include <torch/torch.h>

#include <core/rope.h>

int main() {
  // Basic CUDA smoke test: create tiny Q/K and run RoPE in-place.
  if (!torch::cuda::is_available()) {
    std::fprintf(stderr, "SKIP: CUDA not available\n");
    return 0;
  }

  const int device_index = 0;
  const int64_t head_dim = 8;
  const int64_t seq_len = 4;

  qwen::RopeTables tables =
      qwen::precompute_cos_sin(/*max_seq_len=*/128,
                               /*head_dim=*/head_dim,
                               /*theta=*/10000.0f,
                               /*dtype=*/torch::kFloat16,
                               /*device_index=*/device_index);

  auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, device_index);
  auto q = torch::zeros({1, 1, seq_len, head_dim}, opts);
  auto k = torch::zeros({1, 1, seq_len, head_dim}, opts);

  qwen::apply_rope_inplace(q, k, tables, /*pos_offset=*/0);

  CHECK_TRUE(torch::isfinite(q).all().item<bool>());
  CHECK_TRUE(torch::isfinite(k).all().item<bool>());
  return 0;
}
