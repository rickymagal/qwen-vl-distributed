#include <torch/torch.h>
#include <iostream>
#include "core/config.h"
#include "model/model_stage.h"

using namespace qwen;

int main(int argc, char** argv) {
  ModelConfig cfg;
  cfg.stage_id = 3;
  cfg.stage_count = 4;
  cfg.layer_start = 48;
  cfg.layer_end = cfg.num_hidden_layers; // last blocks
  cfg.device_index = 0;

  torch::Device device(torch::kCUDA, cfg.device_index);
  torch::cuda::setDevice(cfg.device_index);

  ModelStage stage(cfg);
  stage->to(device);

  torch::Tensor hidden = torch::randn({1, 16, cfg.hidden_size}, device);

  StageInput in;
  in.hidden_in = hidden;
  in.pos = 32;

  StageOutput out = stage->forward(in);

  if (out.logits.defined()) {
    std::cout << "[stageN] logits shape: "
              << out.logits.sizes() << std::endl;
  } else {
    std::cout << "[stageN] no logits produced" << std::endl;
  }

  return 0;
}
