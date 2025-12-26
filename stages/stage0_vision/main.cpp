#include <torch/torch.h>
#include <iostream>
#include "core/config.h"
#include "model/model_stage.h"

using namespace qwen;

int main(int argc, char** argv) {
  ModelConfig cfg;
  cfg.stage_id = 0;
  cfg.stage_count = 4;
  cfg.layer_start = 0;
  cfg.layer_end = 0; // vision-only stage
  cfg.device_index = 0;

  torch::Device device(torch::kCUDA, cfg.device_index);
  torch::cuda::setDevice(cfg.device_index);

  ModelStage stage(cfg);
  stage->to(device);

  // Dummy vision input (shape placeholder)
  torch::Tensor images = torch::randn({1, 3, 224, 224}, device);

  StageInput in;
  in.images = images;
  in.pos = 0;

  StageOutput out = stage->forward(in);

  std::cout << "[stage0] produced hidden shape: "
            << out.hidden_out.sizes() << std::endl;

  return 0;
}
