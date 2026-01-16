#include "mini_test.h"

#include <torch/torch.h>

#include "core/config.h"
#include "core/tensor_utils.h"
#include "loader/model_loader.h"
#include "loader/weight_loader.h"
#include "model/model_stage.h"

static void fill_param(qwen::MapWeightLoader& wl, const std::string& key, const std::vector<int64_t>& shape) {
  auto t = torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  wl.insert(key, t);
}

int main() {
  SKIP_IF(!torch::cuda::is_available(), "CUDA not available");

  qwen::ModelConfig cfg;
  cfg.vocab_size = 32;
  cfg.hidden_size = 16;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 2;
  cfg.intermediate_size = 32;
  cfg.moe_intermediate_size = 32;
  cfg.num_experts = 2;
  cfg.top_k = 1;
  cfg.use_moe = true;
  cfg.moe_layer_freq = 1;
  cfg.rms_norm_eps = 1e-6f;
  cfg.use_qk_norm = true;
  cfg.stage_id = 0;
  cfg.stage_count = 1;
  cfg.layer_start = 0;
  cfg.layer_end = 2;

  const int64_t head_dim = cfg.hidden_size / cfg.num_attention_heads;
  const int64_t kv_dim = cfg.num_key_value_heads * head_dim;
  const int64_t E = cfg.num_experts;
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.moe_intermediate_size;

  qwen::ModelStage stage(cfg);
  stage->to(torch::Device(torch::kCUDA, 0));
  stage->eval();

  qwen::MapWeightLoader wl;
  const std::string lm_prefix = "model.language_model";

  fill_param(wl, lm_prefix + ".embed_tokens.weight", {cfg.vocab_size, cfg.hidden_size});
  fill_param(wl, lm_prefix + ".norm.weight", {cfg.hidden_size});
  fill_param(wl, "lm_head.weight", {cfg.vocab_size, cfg.hidden_size});

  for (int i = 0; i < cfg.num_hidden_layers; ++i) {
    const std::string base = lm_prefix + ".layers." + std::to_string(i);
    fill_param(wl, base + ".input_layernorm.weight", {H});
    fill_param(wl, base + ".post_attention_layernorm.weight", {H});
    fill_param(wl, base + ".self_attn.q_proj.weight", {H, H});
    fill_param(wl, base + ".self_attn.k_proj.weight", {kv_dim, H});
    fill_param(wl, base + ".self_attn.v_proj.weight", {kv_dim, H});
    fill_param(wl, base + ".self_attn.o_proj.weight", {H, H});
    fill_param(wl, base + ".self_attn.q_norm.weight", {head_dim});
    fill_param(wl, base + ".self_attn.k_norm.weight", {head_dim});

    // MoE: gate + combined gate_up + down
    fill_param(wl, base + ".mlp.gate.weight", {E, H});
    fill_param(wl, base + ".mlp.experts.gate_up_proj", {E, 2 * I, H});
    fill_param(wl, base + ".mlp.experts.down_proj", {E, H, I});
  }

  qwen::LoadReport rep;
  qwen::LoadOptions opts;
  opts.strict = true;
  opts.load_vision = false;
  CHECK_TRUE(qwen::load_stage_weights(stage, wl, cfg, &rep, opts));
  CHECK_EQ(rep.missing, 0);
  CHECK_EQ(rep.mismatched, 0);
  CHECK_TRUE(rep.loaded > 0);

  // Missing key behavior (non-strict).
  qwen::MapWeightLoader wl2;
  const std::string missing_key = "model.language_model.layers.0.input_layernorm.weight";
  for (const auto& k : wl.list_keys()) {
    if (k == missing_key) continue;
    wl2.insert(k, wl.get(k));
  }
  qwen::LoadReport rep2;
  qwen::LoadOptions opts2;
  opts2.strict = false;
  CHECK_TRUE(qwen::load_stage_weights(stage, wl2, cfg, &rep2, opts2));
  CHECK_TRUE(rep2.missing > 0);

  return 0;
}
