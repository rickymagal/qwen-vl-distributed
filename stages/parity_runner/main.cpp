#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "core/config.h"
#include "core/hf_config.h"
#include "core/sharding.h"
#include "loader/model_loader.h"
#include "loader/pt_weight_loader.h"
#include "model/model_stage.h"

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
               "parity_runner usage:\n"
               "  --hf-config <path>\n"
               "  --weights <weights.pt>\n"
               "  --out <output.pt>\n"
               "  [--report <report.json>]\n"
               "  [--input-ids <input_ids.pt>]\n"
               "  [--images <images.pt>]\n"
               "  [--device <cuda_device_index>]\n"
               "  [--num-stages <N>]\n"
               "  [--stage-idx <i>]\n"
               "  [--layer-begin <L>]\n"
               "  [--layer-end <R>]\n");
}

int main(int argc, char** argv) {
  const std::string hf_path = arg_str(argc, argv, "--hf-config", "");
  const std::string weights_path = arg_str(argc, argv, "--weights", "");
  const std::string out_path = arg_str(argc, argv, "--out", "");
  const std::string report_path = arg_str(argc, argv, "--report", "");
  if (hf_path.empty() || weights_path.empty() || out_path.empty()) {
    usage();
    return 2;
  }

  const std::string input_ids_path = arg_str(argc, argv, "--input-ids", "");
  const std::string images_path = arg_str(argc, argv, "--images", "");

  const int64_t device_index = arg_i64(argc, argv, "--device", 0);
  const int64_t num_stages = arg_i64(argc, argv, "--num-stages", 1);
  const int64_t stage_idx = arg_i64(argc, argv, "--stage-idx", 0);
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

  qwen::PtWeightLoader pt(weights_path);
  pt.load();
  qwen::MapWeightLoader wl;
  for (const auto& kv : pt.weights()) {
    wl.insert(kv.first, kv.second);
  }

  qwen::ModelStage stage(cfg);
  stage->to(torch::Device(torch::kCUDA, (int)device_index));
  stage->eval();

  qwen::LoadReport rep;
  qwen::LoadOptions opts;
  opts.strict = true;
  opts.load_vision = false;
  qwen::load_stage_weights(stage, wl, cfg, &rep, opts);

  std::fprintf(stderr,
               "[parity_runner] loaded=%lld missing=%lld mismatched=%lld\n",
               (long long)rep.loaded,
               (long long)rep.missing,
               (long long)rep.mismatched);

  if (!report_path.empty()) {
    auto extra = qwen::diff_unused_keys(wl, rep.used_keys);
    std::ofstream os(report_path);
    os << "{\n";
    os << "  \"loaded\": " << rep.loaded << ",\n";
    os << "  \"missing\": " << rep.missing << ",\n";
    os << "  \"mismatched\": " << rep.mismatched << ",\n";
    os << "  \"extra\": " << extra.size() << "\n";
    os << "}\n";
  }

  qwen::StageInput in;
  if (!input_ids_path.empty()) {
    torch::Tensor input_ids;
    torch::load(input_ids, input_ids_path);
    in.input_ids = input_ids.to(torch::kCUDA, (int)device_index);
  } else if (cfg.vocab_size > 0) {
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::Device(torch::kCUDA, (int)device_index));
    in.input_ids = torch::randint(0, cfg.vocab_size, {1, 8}, opts_i64);
  }

  if (!images_path.empty()) {
    torch::Tensor images;
    torch::load(images, images_path);
    in.images = images.to(torch::kCUDA, (int)device_index);
  }

  in.pos = 0;
  qwen::StageOutput out = stage->forward(in);
  torch::Tensor to_save = out.logits.defined() ? out.logits : out.hidden_out;
  torch::save(to_save, out_path);

  return 0;
}
