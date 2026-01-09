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
  const std::string k = std::string("--") + key;
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == k) return argv[i + 1];
  }
  return def;
}

static int64_t arg_i64(int argc, char** argv, const char* key, int64_t def) {
  const char* s = arg_str(argc, argv, key, nullptr);
  if (!s) return def;
  return std::strtoll(s, nullptr, 10);
}

static void usage(const char* argv0) {
  std::fprintf(
      stderr,
      "Usage: %s --hf-config <path> [--device <cuda:0>] [--num-stages N] [--stage-idx I] "
      "[--layer-begin L] [--layer-end R]\n",
      argv0);
}

int main(int argc, char** argv) {
  if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
    usage(argv[0]);
    return 0;
  }

  const char* hf_path = arg_str(argc, argv, "hf-config", nullptr);
  if (!hf_path) {
    std::fprintf(stderr, "missing --hf-config\n");
    usage(argv[0]);
    return 2;
  }

  const std::string device_s = arg_str(argc, argv, "device", "cuda:0");
  const int64_t num_stages = arg_i64(argc, argv, "num-stages", 4);
  const int64_t stage_idx = arg_i64(argc, argv, "stage-idx", 3);
  const int64_t override_begin = arg_i64(argc, argv, "layer-begin", -1);
  const int64_t override_end = arg_i64(argc, argv, "layer-end", -1);

  if (device_s.rfind("cuda", 0) == 0 && !torch::cuda::is_available()) {
    std::fprintf(stderr, "CUDA not available.\n");
    return 3;
  }

  qwen::ModelConfig base = qwen::load_hf_config_json(hf_path);

  qwen::ShardingPlan plan = qwen::make_plan_even_layers(base, (int32_t)num_stages, std::vector<int>{});
  qwen::ShardSpec spec = plan.stages.at((size_t)stage_idx);
  if (override_begin >= 0 && override_end >= 0) {
    spec.layer_start = (int32_t)override_begin;
    spec.layer_end = (int32_t)override_end;
  }

  qwen::ModelConfig cfg = qwen::config_for_stage(base, spec);

  std::fprintf(
      stderr,
      "[stage3] stage_idx=%lld/%lld layers=[%d,%d) hidden=%d heads=%d kv_heads=%d\n",
      (long long)stage_idx,
      (long long)num_stages,
      (int)cfg.layer_start,
      (int)cfg.layer_end,
      (int)cfg.hidden_size,
      (int)cfg.num_attention_heads,
      (int)cfg.num_key_value_heads);

  torch::Device device(device_s);
  qwen::ModelStage stage(cfg);
  stage->to(device);
  stage->eval();

  // Dummy hidden input for non-embedding stages.
  torch::Tensor hidden = torch::randn(
      {1, 1, cfg.hidden_size},
      torch::TensorOptions().dtype(torch::kFloat16).device(device));

  qwen::StageInput in;
  in.hidden_in = hidden;
  in.pos = 0;

  qwen::StageOutput out = stage->forward(in);
  if (!out.hidden_out.defined()) {
    std::fprintf(stderr, "[stage3] hidden_out is undefined\n");
    return 4;
  }

  const std::string hsz = c10::str(out.hidden_out.sizes());
  std::fprintf(stderr, "[stage3] hidden_out sizes=%s\n", hsz.c_str());

  if (out.logits.defined()) {
    const std::string lsz = c10::str(out.logits.sizes());
    std::fprintf(stderr, "[stage3] logits sizes=%s\n", lsz.c_str());
  } else {
    std::fprintf(stderr, "[stage3] logits undefined (stage may not be final)\n");
  }

  return 0;
}
