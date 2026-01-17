#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "core/config.h"
#include "core/hf_config.h"
#include "core/sharding.h"
#include "loader/model_loader.h"
#include "loader/pt_weight_loader.h"
#include "model/model_stage.h"
#include "runtime/transport.h"

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
               "distributed_parity_stage usage:\n"
               "  --hf-config <path>\n"
               "  --weights <weights.pt>\n"
               "  --num-stages <N>\n"
               "  --stage-idx <i>\n"
               "  [--listen <port>]              (required for non-first stages)\n"
               "  [--next-host <host>]           (required for non-last stages)\n"
               "  [--next-port <port>]           (required for non-last stages)\n"
               "  [--out <output.pt>]            (required for last stage)\n"
               "  [--input-ids <input_ids.pt>]   (first stage only)\n"
               "  [--images <images.pt>]         (first stage only)\n"
               "  [--device <cuda_device_index>]\n"
               "  [--layer-begin <L>]\n"
               "  [--layer-end <R>]\n");
}

int main(int argc, char** argv) {
  const std::string hf_path = arg_str(argc, argv, "--hf-config", "");
  const std::string weights_path = arg_str(argc, argv, "--weights", "");
  const int64_t num_stages = arg_i64(argc, argv, "--num-stages", -1);
  const int64_t stage_idx = arg_i64(argc, argv, "--stage-idx", -1);
  if (hf_path.empty() || weights_path.empty() || num_stages <= 0 || stage_idx < 0) {
    usage();
    return 2;
  }

  const int64_t device_index = arg_i64(argc, argv, "--device", 0);
  const int64_t listen_port = arg_i64(argc, argv, "--listen", -1);
  const std::string next_host = arg_str(argc, argv, "--next-host", "");
  const int64_t next_port = arg_i64(argc, argv, "--next-port", -1);
  const std::string out_path = arg_str(argc, argv, "--out", "");
  const int64_t layer_begin_override = arg_i64(argc, argv, "--layer-begin", -1);
  const int64_t layer_end_override = arg_i64(argc, argv, "--layer-end", -1);

  const bool is_first = (stage_idx == 0);
  const bool is_last = (stage_idx == num_stages - 1);

  if (!is_first && listen_port < 0) {
    std::fprintf(stderr, "error: --listen required for non-first stages\n");
    return 3;
  }
  if (!is_last && (next_host.empty() || next_port < 0)) {
    std::fprintf(stderr, "error: --next-host/--next-port required for non-last stages\n");
    return 3;
  }
  if (is_last && out_path.empty()) {
    std::fprintf(stderr, "error: --out required for last stage\n");
    return 3;
  }

  if (!torch::cuda::is_available()) {
    std::fprintf(stderr, "error: CUDA is not available\n");
    return 4;
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

  qwen::StageInput in;

  if (is_first) {
    const std::string input_ids_path = arg_str(argc, argv, "--input-ids", "");
    const std::string images_path = arg_str(argc, argv, "--images", "");
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
  } else {
    qwen::TcpServer server((int)listen_port);
    qwen::TcpConn conn(server.accept_one());
    qwen::ActivationPacket p = conn.recv_activation();
    in.hidden_in = p.hidden.to(torch::kCUDA, (int)device_index);
    if (p.attn_mask.has_value() && p.attn_mask->defined()) {
      in.attn_mask = p.attn_mask->to(torch::kCUDA, (int)device_index);
    }
    in.pos = p.pos;
  }

  qwen::StageOutput out = stage->forward(in);

  if (is_last) {
    torch::Tensor to_save = out.logits.defined() ? out.logits : out.hidden_out;
    torch::save(to_save, out_path);
    std::fprintf(stderr, "[distributed_parity_stage] saved output -> %s\n", out_path.c_str());
    return 0;
  }

  qwen::ActivationPacket p;
  p.stage_from = (int32_t)stage_idx;
  p.stage_to = (int32_t)(stage_idx + 1);
  p.step = 0;
  p.pos = in.pos;
  p.hidden = out.hidden_out;
  if (out.hidden_out.defined()) {
    qwen::TcpClient client(next_host, (int)next_port);
    client.send_activation(p);
  } else {
    std::fprintf(stderr, "error: hidden_out undefined\n");
    return 5;
  }

  return 0;
}
