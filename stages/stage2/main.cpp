// stages/stage2/main.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "core/config.h"
#include "core/hf_config.h"
#include "core/sharding.h"
#include "core/tensor_utils.h"
#include "model/model_stage.h"

namespace {

static bool has_flag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return true;
  }
  return false;
}

static std::string arg_str(int argc, char** argv, const char* name, const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::string(argv[i + 1]);
  }
  return def;
}

static int64_t arg_i64(int argc, char** argv, const char* name, int64_t def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::strtoll(argv[i + 1], nullptr, 10);
  }
  return def;
}

static void require_file_exists(const std::string& path, const char* what) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good()) {
    std::fprintf(stderr, "error: cannot open %s: '%s'\n", what, path.c_str());
    std::exit(2);
  }
}

struct StateDict {
  c10::Dict<c10::IValue, c10::IValue> d;
  bool has(const std::string& k) const { return d.contains(c10::IValue(k)); }
  torch::Tensor get(const std::string& k) const {
    auto v = d.at(c10::IValue(k));
    if (!v.isTensor()) {
      std::fprintf(stderr, "error: state_dict['%s'] is not a Tensor\n", k.c_str());
      std::exit(3);
    }
    return v.toTensor();
  }
  int64_t size() const { return static_cast<int64_t>(d.size()); }
};

static StateDict load_state_dict(const std::string& weights_path) {
  require_file_exists(weights_path, "weights.pt");
  c10::IValue iv;
  torch::load(iv, weights_path);
  if (!iv.isGenericDict()) {
    std::fprintf(stderr, "error: weights file is not a dict-like object: '%s'\n", weights_path.c_str());
    std::exit(3);
  }
  StateDict sd;
  sd.d = iv.toGenericDict();
  return sd;
}

static bool try_copy_param_from_sd(torch::Tensor param, const StateDict& sd, const std::string& key) {
  if (!sd.has(key)) return false;
  torch::Tensor src = sd.get(key);
  if (src.device() != param.device()) src = src.to(param.device());
  if (src.dtype() != param.dtype()) src = src.to(param.dtype());
  if (src.sizes() != param.sizes()) return false;
  param.copy_(src);
  return true;
}

static std::vector<std::string> hf_key_candidates_for_param(const std::string& param_name) {
  std::vector<std::string> c;

  if (param_name == "embedding.tok_embed.weight") {
    c = {"model.embed_tokens.weight", "model.model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight"};
    return c;
  }
  if (param_name == "ln_f.weight") {
    c = {"model.norm.weight", "model.model.norm.weight", "transformer.ln_f.weight"};
    return c;
  }
  if (param_name == "lm_head.weight") {
    c = {"lm_head.weight", "model.lm_head.weight", "model.model.lm_head.weight"};
    return c;
  }

  const std::string pfx = "block_";
  if (param_name.rfind(pfx, 0) != 0) return c;

  size_t idx_end = param_name.find('.');
  if (idx_end == std::string::npos) return c;
  const std::string blk = param_name.substr(0, idx_end);
  const std::string rest = param_name.substr(idx_end + 1);

  const char* s = blk.c_str() + pfx.size();
  char* e = nullptr;
  long li = std::strtol(s, &e, 10);
  if (!e || *e != '\0' || li < 0) return c;
  const int64_t layer = static_cast<int64_t>(li);

  auto fmt = [&](const std::string& suffix) {
    return std::string("model.layers.") + std::to_string(layer) + "." + suffix;
  };

  if (rest == "ln1.weight") return {fmt("input_layernorm.weight")};
  if (rest == "ln1.bias")   return {fmt("input_layernorm.bias")};
  if (rest == "ln2.weight") return {fmt("post_attention_layernorm.weight")};
  if (rest == "ln2.bias")   return {fmt("post_attention_layernorm.bias")};

  if (rest == "attn.wq.weight") return {fmt("self_attn.q_proj.weight")};
  if (rest == "attn.wq.bias")   return {fmt("self_attn.q_proj.bias")};
  if (rest == "attn.wk.weight") return {fmt("self_attn.k_proj.weight")};
  if (rest == "attn.wk.bias")   return {fmt("self_attn.k_proj.bias")};
  if (rest == "attn.wv.weight") return {fmt("self_attn.v_proj.weight")};
  if (rest == "attn.wv.bias")   return {fmt("self_attn.v_proj.bias")};
  if (rest == "attn.wo.weight") return {fmt("self_attn.o_proj.weight")};
  if (rest == "attn.wo.bias")   return {fmt("self_attn.o_proj.bias")};

  if (rest == "attn.q_norm.weight") return {fmt("self_attn.q_norm.weight"), fmt("self_attn.q_layernorm.weight")};
  if (rest == "attn.k_norm.weight") return {fmt("self_attn.k_norm.weight"), fmt("self_attn.k_layernorm.weight")};

  if (rest == "moe.router.weight") return {fmt("mlp.gate.weight"), fmt("mlp.router.weight"), fmt("mlp.gate_proj.weight")};
  if (rest == "moe.router.bias")   return {fmt("mlp.gate.bias"),   fmt("mlp.router.bias"),   fmt("mlp.gate_proj.bias")};

  const std::string ex_pfx = "moe.experts.";
  if (rest.rfind(ex_pfx, 0) == 0) {
    std::string tail = rest.substr(ex_pfx.size());
    size_t dot = tail.find('.');
    if (dot != std::string::npos) {
      const std::string e_str = tail.substr(0, dot);
      const std::string e_rest = tail.substr(dot + 1);

      char* ee = nullptr;
      long ei = std::strtol(e_str.c_str(), &ee, 10);
      if (ee && *ee == '\0' && ei >= 0) {
        const int64_t expert = static_cast<int64_t>(ei);
        auto fmt_e = [&](const std::string& suffix) {
          return std::string("model.layers.") + std::to_string(layer) + ".mlp.experts." +
                 std::to_string(expert) + "." + suffix;
        };

        if (e_rest == "fc1.weight") return {fmt_e("fc1.weight"), fmt_e("w1.weight"), fmt_e("gate_proj.weight"), fmt_e("up_proj.weight")};
        if (e_rest == "fc1.bias")   return {fmt_e("fc1.bias"),   fmt_e("w1.bias"),   fmt_e("gate_proj.bias"),   fmt_e("up_proj.bias")};
        if (e_rest == "fc2.weight") return {fmt_e("fc2.weight"), fmt_e("w2.weight"), fmt_e("down_proj.weight")};
        if (e_rest == "fc2.bias")   return {fmt_e("fc2.bias"),   fmt_e("w2.bias"),   fmt_e("down_proj.bias")};
      }
    }
  }

  return c;
}

static void apply_weights_best_effort(qwen::ModelStage& stage, const StateDict& sd) {
  torch::NoGradGuard ng;
  int64_t loaded = 0;
  int64_t missing = 0;

  auto params = stage->named_parameters(/*recurse=*/true);
  for (const auto& it : params) {
    const std::string name = it.key();
    torch::Tensor p = it.value();

    auto cand = hf_key_candidates_for_param(name);
    bool ok = false;
    for (const auto& k : cand) {
      if (try_copy_param_from_sd(p, sd, k)) { ok = true; break; }
    }

    if (ok) ++loaded;
    else ++missing;
  }

  std::fprintf(stderr, "weights: state_dict tensors=%lld, params_loaded=%lld, params_unmatched=%lld\n",
               (long long)sd.size(), (long long)loaded, (long long)missing);
}

} // namespace

int main(int argc, char** argv) {
  const int64_t stage_idx = 2;

  if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
    std::printf(
      "stage2\n"
      "  --hf-config  /path/hf_config.json   (required)\n"
      "  --weights    /path/weights.pt       (required)\n"
      "  --device     <cuda_device_index>    (default 0)\n"
      "  --num-stages <N>                    (default 4)\n"
      "  --layer-begin <L> --layer-end <R>   (optional override)\n"
    );
    return 0;
  }

  const std::string hf_path = arg_str(argc, argv, "--hf-config", "");
  const std::string weights_path = arg_str(argc, argv, "--weights", "");
  const int64_t device_index = arg_i64(argc, argv, "--device", 0);
  const int64_t num_stages = arg_i64(argc, argv, "--num-stages", 4);

  if (hf_path.empty()) {
    std::fprintf(stderr, "error: --hf-config is required\n");
    return 2;
  }
  if (weights_path.empty()) {
    std::fprintf(stderr, "error: --weights is required\n");
    return 2;
  }

  qwen::HfConfig hf = qwen::load_hf_config_json(hf_path);
  qwen::ModelConfig cfg = qwen::to_model_config(hf);

  int64_t layer_begin = -1;
  int64_t layer_end = -1;
  if (has_flag(argc, argv, "--layer-begin")) layer_begin = arg_i64(argc, argv, "--layer-begin", -1);
  if (has_flag(argc, argv, "--layer-end"))   layer_end   = arg_i64(argc, argv, "--layer-end", -1);

  qwen::ShardBoundary sb;
  if (layer_begin >= 0 && layer_end >= 0) {
    sb.stage_idx = (int32_t)stage_idx;
    sb.num_stages = (int32_t)num_stages;
    sb.layer_begin = (int32_t)layer_begin;
    sb.layer_end = (int32_t)layer_end;
  } else {
    sb = qwen::choose_shard_boundaries(cfg.num_hidden_layers, (int32_t)stage_idx, (int32_t)num_stages);
  }

  std::fprintf(stderr,
    "cfg: hidden=%d layers=%d heads=%d vocab=%d moe=%d experts=%d topk=%d\n",
    cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.vocab_size,
    (int)cfg.use_moe, cfg.num_experts, cfg.top_k
  );
  std::fprintf(stderr,
    "shard: stage=%d/%d layers=[%d, %d)\n",
    sb.stage_idx, sb.num_stages, sb.layer_begin, sb.layer_end
  );

  StateDict sd = load_state_dict(weights_path);

  qwen::ModelStage stage(cfg);
  apply_weights_best_effort(stage, sd);

  if (!torch::cuda::is_available()) {
    std::fprintf(stderr, "error: CUDA is not available in this build/runtime\n");
    return 2;
  }
  torch::Device dev(torch::kCUDA, (int)device_index);
  stage->to(dev);

  torch::NoGradGuard ng;
  qwen::StageInput in;
  in.pos = 0;
  in.hidden = torch::zeros({1, 1, cfg.hidden_size}, torch::dtype(torch::kFloat32).device(dev));

  auto out = stage->forward(in);

  qwen::require(out.hidden_out.defined(), "stage2: hidden_out undefined");
  qwen::require_cuda(out.hidden_out, "stage2: hidden_out must be CUDA");

  std::printf("stage2 sanity forward ok (hidden_out=%s)\n", out.hidden_out.sizes().str().c_str());
  return 0;
}
