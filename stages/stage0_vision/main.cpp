#include <cstdlib>
#include <iostream>
#include <string>

#include "core/tensor_utils.h"
#include "vision/vision_encoder.h"

static int arg_int(int argc, char** argv, const char* key, int defv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
  }
  return defv;
}

static qwen::ModelConfig make_cfg(int argc, char** argv) {
  qwen::ModelConfig cfg;
  cfg.device_index = arg_int(argc, argv, "--device", 0);
  return cfg;
}

int main(int argc, char** argv) {
  qwen::ModelConfig cfg = make_cfg(argc, argv);

  qwen::VisionEncoder ve(cfg);

  auto images = torch::zeros({1, 3, 224, 224}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
  auto out = ve->forward(images);

  qwen::require(out.defined(), "stage0_vision: output undefined");
  std::cout << "stage0_vision ok\n";
  return 0;
}
