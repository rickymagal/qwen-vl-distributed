#include <torch/torch.h>

#include "core/config.h"
#include "model/embedding.h"

int main() {
  if (!torch::cuda::is_available()) {
    // Skip on systems without CUDA.
    return 0;
  }

  qwen::ModelConfig cfg;
  cfg.vocab_size = 1000;
  cfg.hidden_size = 64;
  cfg.max_seq_len = 16;
  cfg.dtype = "f16";

  qwen::Embedding emb(cfg);
  emb->to(torch::Device(torch::kCUDA, 0));

  torch::Tensor input_ids = torch::randint(
      0, cfg.vocab_size, {2, 5}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

  torch::Tensor out = emb->forward(input_ids);

  if (out.sizes() != torch::IntArrayRef({2, 5, cfg.hidden_size})) {
    throw std::runtime_error("unexpected embedding output shape");
  }

  return 0;
}
