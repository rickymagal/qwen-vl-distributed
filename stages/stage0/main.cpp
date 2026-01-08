// stages/stage0/main.cpp
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

  bool has(const std::string& k) const {
    return d.contains(c10::IValue(k));
  }

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

  // Basic validation: keys must be strings and values must be tensors (best-effort).
  int64_t checked = 0;
  for (const auto& it : sd.d) {
    if (checked >= 8) break;
    if (!it.key().isString()) {
      std::fprintf(stderr, "error: state_dict contains non-string keys\n");
      std::exit(3);
    }
    if (!it.value().isTensor()) {
      std::fprintf(stderr, "warning: state_dict['%s'] is not a tensor\n", it.key().toStringRef().c_str());
    }
    ++checked;
  }

  return sd;
}

static bool try_copy_param_from_sd(torch::Tensor param, const StateDict& sd, const std::string& key) {
  if (!sd.has(key)) return false;

  torch::Tensor src = sd.get(key);
  auto want_dev = param.device();
  auto want_dtype = param.dtype();

  if (src.device() != want_dev) src = src.to(want_dev);
  if (src.dtype() != want_dtype) src = src.to(want_dtype);

  if (src.sizes() != param.sizes()) {
    std::fprintf(stderr, "warning: size mismatch for key '%s': src=%s param=%s\n",
                 key.c_str(), src.sizes().str().c_str(), param.sizes().str().c_str());
    return false;
  }

  param.copy_(src);
  return true;
}

static std::vector<std::string> hf_key_candidates_for_param(const std::string& param_name) {
  std::vector<std::string> c;

  // Top-level
  if (param_name == "embedding.tok_embed.weight") {
    c.push_back("model.embed_tokens.weight");
    c.push_back("model.model.embed_tokens.weight");
    c.push_back("embed_tokens.weight");
    c.push_back("transformer.wte.weight");
    return c;
  }
  if (param_name == "ln_f.weight") {
    c.push_back("model.norm.weight");
    c.push_back("model.model.norm.weight");
    c.push_back("transformer.ln_f.weight");
    return c;
  }
  if (param_name == "lm_head.weight") {
    c.push_back("lm_head.weight");
    c.push_back("model.lm_head.weight");
    c.push_back("model.model.lm_head.weight");
    return c;
  }

  // Blocks: block_<i>.*
  const std::string pfx = "block_";
  if (param_name.rfind(pfx, 0) != 0) return c;

  // Parse block index
  size_t idx_end = param_name.find('.');
  if (idx_end == std::string::npos) return c;
  const std::string blk = param_name.substr(0, idx_end); // block_<i>
  const std::string rest = param_name.substr(idx_end + 1);

  const char* s = blk.c_str() + pfx.size();
  char* e = nullptr;
  long li = std::strtol(s, &e, 10);
  if (!e || *e != '\0' || li < 0) return c;
  const int64_t layer = static_cast<int64_t>(li);

  auto fmt = [&](const std::string& suffix) {
    return std::string("model.layers.") + std::to_string(layer) + "." + suffix;
  };

  // Layer norms
  if (rest == "ln1.weight") { c.push_back(fmt("input_layernorm.weight")); return c; }
  if (rest == "ln1.bias")   { c.push_back(fmt("input_layernorm.bias"));   return c; }
  if (rest == "ln2.weight") { c.push_back(fmt("post_attention_layernorm.weight")); return c; }
  if (rest == "ln2.bias")   { c.push_back(fmt("post_attention_layernorm.bias"));   return c; }

  // Attention projections
  if (rest == "attn.wq.weight") { c.push_back(fmt("self_attn.q_proj.weight")); return c; }
  if (rest == "attn.wq.bias")   { c.push_back(fmt("self_attn.q_proj.bias"));   return c; }
  if (rest == "attn.wk.weight") { c.push_back(fmt("self_attn.k_proj.weight")); return c; }
  if (rest == "attn.wk.bias")   { c.push_back(fmt("self_attn.k_proj.bias"));   return c; }
  if (rest == "attn.wv.weight") { c.push_back(fmt("self_attn.v_proj.weight")); return c; }
  if (rest == "attn.wv.bias")   { c.push_back(fmt("self_attn.v_proj.bias"));   return c; }
  if (rest == "attn.wo.weight") { c.push_back(fmt("self_attn.o_proj.weight")); return c; }
  if (rest == "attn.wo.bias")   { c.push_back(fmt("self_attn.o_proj.bias"));   return c; }

  // Optional per-head norms (names vary across repos)
  if (rest == "attn.q_norm.weight") {
    c.push_back(fmt("self_attn.q_norm.weight"));
    c.push_back(fmt("self_attn.q_layernorm.weight"));
    return c;
  }
  if (rest == "attn.k_norm.weight") {
    c.push_back(fmt("self_attn.k_norm.weight"));
    c.push_back(fmt("self_attn.k_layernorm.weight"));
    return c;
  }

  // MoE router
  if (rest == "moe.router.weight") {
    c.push_back(fmt("mlp.gate.weight"));
    c.push_back(fmt("mlp.router.weight"));
    c.push_back(fmt("mlp.gate_proj.weight"));
    return c;
  }
  if (rest == "moe.router.bias") {
    c.push_back(fmt("mlp.gate.bias"));
    c.push_back(fmt("mlp.router.bias"));
    c.push_back(fmt("mlp.gate_proj.bias"));
    return c;
  }

  // Experts: moe.experts.<e>.{fc1,fc2}.{weight,bias}
  // We generate a few common key patterns; exact Qwen naming may differ.
  const std::string ex_pfx = "moe.experts.";
  if (rest.rfind(ex_pfx, 0) == 0) {
    // rest: moe.experts.<e>.fc1.weight
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

        if (e_rest == "fc1.weight") {
          c.push_back(fmt_e("fc1.weight"));
          c.push_back(fmt_e("w1.weight"));
          c.push_back(fmt_e("gate_proj.weight"));
          c.push_back(fmt_e("up_proj.weight"));
          return c;
        }
        if (e_rest == "fc1.bias") {
          c.push_back(fmt_e("fc1.bias"));
          c.push_back(fmt_e("w1.bias"));
          c.push_back(fmt_e("gate_proj.bias"));
          c.push_back(fmt_e("up_proj.bias"));
          return c;
        }
        if (e_rest == "fc2.weight") {
          c.push_back(fmt_e("fc2.weight"));
          c.push_back(fmt_e("w2.weight"));
          c.push_back(fmt_e("down_proj.weight"));
          return c;
        }
        if (e_rest == "fc2.bias") {
          c.push_back(fmt_e("fc2.bias"));
          c.push_back(fmt_e("w2.bias"));
          c.push_back(fmt_e("down_proj.bias"));
          return c;
        }
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

    if (ok) {
      ++loaded;
    } else {
      ++missing;
      // Keep noise low; print only a few misses.
      if (missing <= 20) {
        std::fprintf(stderr, "warning: no HF key matched for param '%s'\n", name.c_str());
      }
    }
  }

  std::fprintf(stderr, "weights: state_dict tensors=%lld, params_loaded=%lld, params_unmatched=%lld\n",
               (long long)sd.size(), (long long)loaded, (long long)missing);
}

} // namespace

int main(int argc, char** argv) {
  const int64_t stage_idx = 0;

  if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
    std::printf(
      "stage0\n"
      "  --hf-config  /path/hf_config.json   (required)\n"
      "  --weights    /path/weights.pt       (required)\n"
      "  --device     <cuda_device_index>    (default 0)\n"
      "  --num-stages <N>                    (default 4)\n"
      "  --layer-begin <L> --layer-end <R>   (optional override)\n"
      "\n"
      "Demonstrates:\n"
      "  - Load HF config -> ModelConfig\n"
      "  - Compute shard boundaries\n"
      "  - Load weights.pt (state_dict) and best-effort map to parameters\n"
      "  - Move model to CUDA device\n"
      "  - Run a 1-token sanity forward\n"
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

  // Load weights and apply best-effort mapping on CPU first (fast failure if file is invalid).
  StateDict sd = load_state_dict(weights_path);

  qwen::ModelStage stage(cfg);
  apply_weights_best_effort(stage, sd);

  // Move model to CUDA.
  if (!torch::cuda::is_available()) {
    std::fprintf(stderr, "error: CUDA is not available in this build/runtime\n");
    return 2;
  }
  torch::Device dev(torch::kCUDA, (int)device_index);
  stage->to(dev);

  // Sanity forward: 1 token.
  torch::NoGradGuard ng;
  qwen::StageInput in;
  in.pos = 0;
  in.input_ids = torch::zeros({1, 1}, torch::dtype(torch::kInt64).device(dev));
  // Note: input_ids must be CUDA (enforced by Embedding::forward).

  auto out = stage->forward(in);

  qwen::require(out.hidden_out.defined(), "stage0: hidden_out undefined");
  qwen::require_cuda(out.hidden_out, "stage0: hidden_out must be CUDA");

  std::printf("stage0 sanity forward ok (hidden_out=%s)\n", out.hidden_out.sizes().str().c_str());
  return 0;
}
