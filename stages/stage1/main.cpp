#include <cstdlib>
#include <iostream>
#include <string>

#include "core/tensor_utils.h"
#include "runtime/pipeline_stage.h"

static std::string arg_str(int argc, char** argv, const char* key, const char* defv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return argv[i + 1];
  }
  return defv;
}

static int arg_int(int argc, char** argv, const char* key, int defv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
  }
  return defv;
}

static qwen::ModelConfig make_cfg(int argc, char** argv) {
  qwen::ModelConfig cfg;
  cfg.model_id = arg_str(argc, argv, "--model", "");
  cfg.device_index = arg_int(argc, argv, "--device", 0);
  cfg.vocab_size = arg_int(argc, argv, "--vocab", 0);
  cfg.hidden_size = arg_int(argc, argv, "--hidden", 0);
  cfg.num_hidden_layers = arg_int(argc, argv, "--layers", 0);
  cfg.num_attention_heads = arg_int(argc, argv, "--heads", 0);
  cfg.use_moe = arg_int(argc, argv, "--moe", 0) != 0;
  return cfg;
}

int main(int argc, char** argv) {
  qwen::ModelConfig cfg = make_cfg(argc, argv);

  qwen::PipelineStage ps(cfg);

  qwen::ActivationPacket p;
  p.version = 1;
  p.stage_from = 0;
  p.stage_to = 1;
  p.pos = 0;
  p.hidden = torch::zeros({1, 1, cfg.hidden_size}, torch::dtype(torch::kFloat16).device(torch::kCUDA));

  auto out = ps.run_from_activation(p, cfg.device_index);

  qwen::require(out.hidden_out.defined(), "stage1: hidden_out undefined");
  std::cout << "stage1 ok\n";
  return 0;
}
