// tests/test_smoke_forward.cu
#include <cstdio>

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "core/tensor_utils.h"
#include "model/model_stage.h"

__global__ void noop() {}

int main() {
  torch::NoGradGuard ng;

  qwen::ModelConfig cfg;
  cfg.vocab_size = 32000;
  cfg.hidden_size = 1024;
  cfg.num_hidden_layers = 1;
  cfg.num_attention_heads = 16;
  cfg.num_key_value_heads = 16;
  cfg.use_moe = false;

  qwen::ModelStage stage(cfg);
  stage->eval();

  const int device_index = 0;
  stage->to(torch::Device(torch::kCUDA, device_index));

  qwen::StageInput in;
  in.pos = 0;
  in.input_ids = torch::zeros({1, 1},
                              torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_index));

  auto out = stage->forward(in);

  qwen::require(out.hidden_out.defined(), "smoke: hidden_out undefined");
  qwen::require_cuda(out.hidden_out, "smoke: hidden_out must be CUDA");

  noop<<<1, 1>>>();
  cudaError_t st = cudaDeviceSynchronize();
  qwen::require(st == cudaSuccess, "smoke: cudaDeviceSynchronize failed");

  std::printf("smoke forward cuda ok\n");
  return 0;
}
