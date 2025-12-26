#include <torch/torch.h>
#include <iostream>
#include "core/config.h"
#include "model/model_stage.h"

using namespace qwen;

int main(int argc, char** argv) {
  ModelConfig cfg;
  cfg.stage_id = 2;
  cfg.stage_count = 4;
  cfg.layer_start = 24;
  cfg.layer_end = 48; // example block range
  cfg.device_index = 0;

  torch::Device device(torch::kCUDA, cfg.device_index);
  torch::cuda::setDevice(cfg.device_index);

  ModelStage stage(cfg);
  stage->to(device);

  torch::Tensor hidden = torch::randn({1, 16, cfg.hidden_size}, device);

  StageInput in;
  in.hidden_in = hidden;
  in.pos = 16;

  StageOutput out = stage->forward(in);

  std::cout << "[stage2] produced hidden shape: "
            << out.hidden_out.sizes() << std::endl;

  return 0;
}
