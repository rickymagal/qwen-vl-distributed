#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "core/config.h"
#include "core/hf_config.h"
#include "core/sharding.h"
#include "model/model_stage.h"

static bool has_flag(int argc, char** argv, const char* flag) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == flag) return true;
  }
  return false;
}

static const char* arg_str(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return argv[i + 1];
  }
  return def;
}

static int64_t arg_i64(int argc, char** argv, const char* key, int64_t def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return std::stoll(argv[i + 1]);
  }
  return def;
}

static void usage() {
  std::fprintf(stderr,
               "stage2 usage:\n"
               "  --hf-config <path>\n"
               "  [--device <cuda_device_index>]\n"
               "  [--num-stages <N>]\n"
               "  [--stage-idx <i>]\n"
               "  [--layer-begin <L>]\n"
               "  [--layer-end <R>]\n");
}

int main(int argc, char** argv) {
  if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
    usage();
    return 0;
  }

  const std::string hf_path = arg_str(argc, argv, "--hf-config", "");
  if (hf_path.empty()) {
    std::fprintf(stderr, "error: missing --hf-config\n");
    usage();
    return 2;
  }

  const int64_t device_index = arg_i64(argc, argv, "--device", 0);
  const int64_t num_stages = arg_i64(argc, argv, "--num-stages", 1);
  const int64_t stage_idx = arg_i64(argc, argv, "--stage-idx", 2);
  const int64_t layer_begin_override = arg_i64(argc, argv, "--layer-begin", -1);
  const int64_t layer_end_override = arg_i64(argc, argv, "--layer-end", -1);

  if (!torch::cuda::is_available()) {
    std::fprintf(stderr, "error: CUDA is not available\n");
    return 3;
  }

  qwen::ModelConfig base_cfg = qwen::load_hf_config_json(hf_path);

  qwen::ShardingPlan plan = qwen::make_plan_even_layers(base_cfg, (int32_t)num_stages, std::vector<int>{});
  qwen::ShardSpec spec = plan.stages.at((size_t)stage_idx);
  if (layer_begin_override >= 0) spec.layer_start = (int32_t)layer_begin_override;
  if (layer_end_override >= 0) spec.layer_end = (int32_t)layer_end_override;

  qwen::ModelConfig cfg = qwen::config_for_stage(base_cfg, spec);

  std::fprintf(stderr,
               "[stage2] device=%lld stages=%lld idx=%lld layers=[%d,%d)\n",
               (long long)device_index,
               (long long)num_stages,
               (long long)stage_idx,
               spec.layer_start,
               spec.layer_end);

  qwen::ModelStage stage(cfg);
  stage->to(torch::Device(torch::kCUDA, (int)device_index));
  stage->eval();

  const int64_t B = 1;
  const int64_t T = 8;
  auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::Device(torch::kCUDA, (int)device_index));
  auto hidden = torch::randn({B, T, cfg.hidden_size}, opts);

  qwen::StageInput in;
  in.hidden_in = hidden;
  in.pos = 0;

  auto out = stage->forward(in);
  if (out.hidden_out.defined()) {
    std::string s = c10::str(out.hidden_out.sizes());
    std::fprintf(stderr, "[stage2] hidden_out: %s\n", s.c_str());
  }
  if (out.logits.defined()) {
    std::string s = c10::str(out.logits.sizes());
    std::fprintf(stderr, "[stage2] logits: %s\n", s.c_str());
  }

  return 0;
}
