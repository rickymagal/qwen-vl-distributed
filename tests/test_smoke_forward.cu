#include <cstdio>
#include <cstdlib>
#include <torch/torch.h>

#include "core/tensor_utils.h"
#include "model/model_stage.h"

__global__ void noop() {}

int main() {
  qwen::ModelConfig cfg;
  cfg.vocab_size = 32000;
  cfg.hidden_size = 1024;
  cfg.num_hidden_layers = 1;
  cfg.num_attention_heads = 16;
  cfg.use_moe = false;

  const bool have_cuda = torch::cuda::is_available();
  qwen::require(have_cuda, "smoke_forward_cuda: CUDA not available");

  const int device_index = 0;
  torch::Device device(torch::kCUDA, device_index);

  qwen::ModelStage stage(cfg);

  // Move the entire stage (including embedding weights) to CUDA.
  stage->to(device);

  qwen::StageInput in;
  in.pos = 0;

  // Embedding path in this repo requires CUDA input_ids.
  in.input_ids = torch::zeros(
      {1, 1},
      torch::TensorOptions().dtype(torch::kInt64).device(device));

  auto out = stage->forward(in);

  qwen::require(out.hidden_out.defined(), "smoke: hidden_out undefined");
  qwen::require_cuda(out.hidden_out, "smoke: hidden_out must be CUDA");
  qwen::require(out.hidden_out.dim() == 3, "smoke: expected [B,T,D] output");

  noop<<<1, 1>>>();
  cudaError_t e = cudaDeviceSynchronize();
  qwen::require(e == cudaSuccess, "smoke: cudaDeviceSynchronize failed");

  std::printf("smoke forward cuda ok\n");
  return 0;
}
