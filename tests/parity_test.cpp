#include <torch/torch.h>
#include <iostream>
#include "core/config.h"
#include "model/model_stage.h"

using namespace qwen;

int main() {
  // Este teste valida apenas invariantes estruturais e determinismo básico.
  ModelConfig cfg;
  cfg.hidden_size = 4096;
  cfg.num_attention_heads = 32;
  cfg.num_hidden_layers = 2;
  cfg.layer_start = 0;
  cfg.layer_end = 2;
  cfg.stage_id = 0;
  cfg.stage_count = 1;
  cfg.device_index = 0;
  cfg.max_batch = 1;
  cfg.max_seq_len = 8;

  torch::cuda::setDevice(cfg.device_index);
  ModelStage stage(cfg);
  stage->to(torch::kCUDA);

  torch::Tensor hidden =
      torch::randn({1, 4, cfg.hidden_size}, torch::device(torch::kCUDA));

  StageInput in;
  in.hidden_in = hidden;
  in.pos = 0;

  auto out1 = stage->forward(in);
  auto out2 = stage->forward(in);

  if (!torch::allclose(out1.hidden_out, out2.hidden_out, 1e-4, 1e-4)) {
    std::cerr << "Parity test failed: outputs differ\n";
    return 1;
  }

  std::cout << "Parity test passed\n";
  return 0;
}
