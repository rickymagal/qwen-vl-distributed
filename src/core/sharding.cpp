// src/core/sharding.cpp
#include "core/sharding.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/tensor_utils.h"

namespace qwen {

static inline std::size_t dtype_bytes_from_cfg_(const ModelConfig& cfg) {
  if (cfg.dtype == "fp16") return 2;
  if (cfg.dtype == "bf16") return 2;
  // Default to 2 bytes for planning if unknown.
  return 2;
}

std::vector<std::pair<int32_t, int32_t>> shard_layers_even(int32_t num_layers, int32_t stage_count) {
  require(num_layers >= 0, "shard_layers_even: num_layers must be >= 0");
  require(stage_count > 0, "shard_layers_even: stage_count must be > 0");

  std::vector<std::pair<int32_t, int32_t>> ranges;
  ranges.reserve(static_cast<std::size_t>(stage_count));

  const int32_t base = (stage_count > 0) ? (num_layers / stage_count) : 0;
  const int32_t rem = (stage_count > 0) ? (num_layers % stage_count) : 0;

  int32_t cur = 0;
  for (int32_t s = 0; s < stage_count; ++s) {
    const int32_t add = base + (s < rem ? 1 : 0);
    const int32_t start = cur;
    const int32_t end = cur + add;
    ranges.push_back({start, end});
    cur = end;
  }

  require(cur == num_layers, "shard_layers_even: internal error (coverage mismatch)");
  return ranges;
}

static std::vector<int32_t> normalize_devices_(int32_t stage_count, const std::vector<int32_t>& device_indices) {
  if (device_indices.empty()) {
    return std::vector<int32_t>(static_cast<std::size_t>(stage_count), 0);
  }
  if (static_cast<int32_t>(device_indices.size()) == stage_count) {
    return device_indices;
  }
  // If user provides a single device, replicate it for all stages.
  if (device_indices.size() == 1 && stage_count > 1) {
    return std::vector<int32_t>(static_cast<std::size_t>(stage_count), device_indices[0]);
  }
  throw std::runtime_error("sharding: device_indices must be empty, size==stage_count, or size==1");
}

ShardingPlan make_plan_even_layers(const ModelConfig& base,
                                   int32_t stage_count,
                                   const std::vector<int32_t>& device_indices) {
  require(stage_count > 0, "make_plan_even_layers: stage_count must be > 0");
  require(base.num_hidden_layers >= 0, "make_plan_even_layers: base.num_hidden_layers must be >= 0");

  const auto ranges = shard_layers_even(base.num_hidden_layers, stage_count);
  const auto devs = normalize_devices_(stage_count, device_indices);

  ShardingPlan plan;
  plan.stages.reserve(static_cast<std::size_t>(stage_count));

  for (int32_t s = 0; s < stage_count; ++s) {
    ShardSpec spec;
    spec.stage_id = s;
    spec.stage_count = stage_count;
    spec.layer_start = ranges[static_cast<std::size_t>(s)].first;
    spec.layer_end = ranges[static_cast<std::size_t>(s)].second;
    spec.device_index = devs[static_cast<std::size_t>(s)];

    spec.est_kv_bytes_per_token = estimate_kv_bytes_per_token(base, spec.layer_start, spec.layer_end);
    spec.est_weight_bytes = estimate_weight_bytes(base, spec.layer_start, spec.layer_end);

    plan.stages.push_back(spec);
  }

  return plan;
}

ShardingPlan make_plan_manual(const ModelConfig& base,
                              const std::vector<std::pair<int32_t, int32_t>>& ranges,
                              const std::vector<int32_t>& device_indices) {
  require(!ranges.empty(), "make_plan_manual: ranges must be non-empty");
  const int32_t stage_count = static_cast<int32_t>(ranges.size());
  const auto devs = normalize_devices_(stage_count, device_indices);

  // Validate coverage (must be contiguous, within [0, num_hidden_layers], and cover all).
  require(base.num_hidden_layers >= 0, "make_plan_manual: base.num_hidden_layers must be >= 0");
  int32_t cur = 0;
  for (int32_t i = 0; i < stage_count; ++i) {
    const auto r = ranges[static_cast<std::size_t>(i)];
    require(r.first == cur, "make_plan_manual: ranges must be contiguous and start at 0");
    require(r.first >= 0 && r.second >= r.first, "make_plan_manual: invalid range");
    require(r.second <= base.num_hidden_layers, "make_plan_manual: range exceeds num_hidden_layers");
    cur = r.second;
  }
  require(cur == base.num_hidden_layers, "make_plan_manual: ranges must cover all layers");

  ShardingPlan plan;
  plan.stages.reserve(static_cast<std::size_t>(stage_count));

  for (int32_t s = 0; s < stage_count; ++s) {
    ShardSpec spec;
    spec.stage_id = s;
    spec.stage_count = stage_count;
    spec.layer_start = ranges[static_cast<std::size_t>(s)].first;
    spec.layer_end = ranges[static_cast<std::size_t>(s)].second;
    spec.device_index = devs[static_cast<std::size_t>(s)];

    spec.est_kv_bytes_per_token = estimate_kv_bytes_per_token(base, spec.layer_start, spec.layer_end);
    spec.est_weight_bytes = estimate_weight_bytes(base, spec.layer_start, spec.layer_end);

    plan.stages.push_back(spec);
  }

  return plan;
}

ModelConfig config_for_stage(const ModelConfig& base, const ShardSpec& s) {
  ModelConfig cfg = base;
  cfg.stage_id = s.stage_id;
  cfg.stage_count = s.stage_count;
  cfg.layer_start = s.layer_start;
  cfg.layer_end = s.layer_end;
  cfg.device_index = s.device_index;
  return cfg;
}

std::size_t estimate_kv_bytes_per_token(const ModelConfig& cfg, int32_t layer_start, int32_t layer_end) {
  require(layer_start >= 0, "estimate_kv_bytes_per_token: layer_start must be >= 0");
  require(layer_end >= layer_start, "estimate_kv_bytes_per_token: layer_end must be >= layer_start");
  require(layer_end <= cfg.num_hidden_layers, "estimate_kv_bytes_per_token: layer_end exceeds num_hidden_layers");

  const int32_t n_layers = layer_end - layer_start;
  if (n_layers == 0) return 0;

  require(cfg.hidden_size > 0, "estimate_kv_bytes_per_token: hidden_size must be > 0");
  require(cfg.num_attention_heads > 0, "estimate_kv_bytes_per_token: num_attention_heads must be > 0");

  const int32_t kv_heads = (cfg.num_key_value_heads > 0) ? cfg.num_key_value_heads : cfg.num_attention_heads;
  const int32_t head_dim = cfg.hidden_size / cfg.num_attention_heads;

  // KV per token per layer per batch:
  // K: [B, kv_heads, head_dim] + V: [B, kv_heads, head_dim]
  // bytes = B * kv_heads * head_dim * 2 * dtype_bytes
  const std::size_t dtype_bytes = dtype_bytes_from_cfg_(cfg);
  const std::size_t per_layer_per_token =
      static_cast<std::size_t>(cfg.max_batch) *
      static_cast<std::size_t>(kv_heads) *
      static_cast<std::size_t>(head_dim) *
      static_cast<std::size_t>(2) *
      dtype_bytes;

  return per_layer_per_token * static_cast<std::size_t>(n_layers);
}

static std::size_t estimate_layer_params_dense_(const ModelConfig& cfg) {
  // Dense transformer block rough param count:
  // - Attention projections: Wq, Wk, Wv, Wo ~ 4 * H * H
  // - MLP: gate, up, down ~ 3 * H * I (I = intermediate_size)
  // - Norms: ~ 2 * H
  require(cfg.hidden_size > 0, "estimate_weight_bytes: hidden_size must be > 0");
  require(cfg.intermediate_size > 0, "estimate_weight_bytes: intermediate_size must be > 0");

  const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
  const std::size_t I = static_cast<std::size_t>(cfg.intermediate_size);

  const std::size_t attn = 4ULL * H * H;
  const std::size_t mlp = 3ULL * H * I;
  const std::size_t norms = 2ULL * H;

  return attn + mlp + norms;
}

static std::size_t estimate_layer_params_moe_(const ModelConfig& cfg) {
  // Very rough MoE block param count:
  // - Router: H * num_experts (small vs experts)
  // - Experts: num_experts * (3 * H * I)
  //
  // This ignores gating/biases/extra norms and assumes expert MLP structure matches dense MLP.
  require(cfg.hidden_size > 0, "estimate_weight_bytes: hidden_size must be > 0");
  require(cfg.intermediate_size > 0, "estimate_weight_bytes: intermediate_size must be > 0");
  require(cfg.num_experts > 0, "estimate_weight_bytes: num_experts must be > 0");

  const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
  const std::size_t I = static_cast<std::size_t>(cfg.intermediate_size);
  const std::size_t E = static_cast<std::size_t>(cfg.num_experts);

  const std::size_t router = H * E;
  const std::size_t experts = E * (3ULL * H * I);

  // Keep attention + norms in addition to MoE experts (typical architectures still have attention).
  const std::size_t attn_and_norms = (4ULL * H * H) + (2ULL * H);

  return attn_and_norms + router + experts;
}

std::size_t estimate_weight_bytes_dense_only(const ModelConfig& cfg, int32_t layer_start, int32_t layer_end) {
  require(layer_start >= 0, "estimate_weight_bytes_dense_only: layer_start must be >= 0");
  require(layer_end >= layer_start, "estimate_weight_bytes_dense_only: layer_end must be >= layer_start");
  require(layer_end <= cfg.num_hidden_layers, "estimate_weight_bytes_dense_only: layer_end exceeds num_hidden_layers");

  const int32_t n_layers = layer_end - layer_start;
  if (n_layers == 0) return 0;

  const std::size_t dtype_bytes = dtype_bytes_from_cfg_(cfg);

  // Add a one-time embedding + lm_head approximation to stage 0 only? We do not know
  // where caller wants to account those, so keep this function strictly "per-layer".
  const std::size_t per_layer_params = estimate_layer_params_dense_(cfg);
  return static_cast<std::size_t>(n_layers) * per_layer_params * dtype_bytes;
}

std::size_t estimate_weight_bytes(const ModelConfig& cfg, int32_t layer_start, int32_t layer_end) {
  require(layer_start >= 0, "estimate_weight_bytes: layer_start must be >= 0");
  require(layer_end >= layer_start, "estimate_weight_bytes: layer_end must be >= layer_start");
  require(layer_end <= cfg.num_hidden_layers, "estimate_weight_bytes: layer_end exceeds num_hidden_layers");

  const int32_t n_layers = layer_end - layer_start;
  if (n_layers == 0) return 0;

  const std::size_t dtype_bytes = dtype_bytes_from_cfg_(cfg);

  std::size_t per_layer_params = 0;
  if (cfg.use_moe) {
    per_layer_params = estimate_layer_params_moe_(cfg);
  } else {
    per_layer_params = estimate_layer_params_dense_(cfg);
  }

  // Add embeddings + final norm/head to the first stage as an approximate "shared" cost.
  // This keeps planning closer to reality without requiring weights.
  std::size_t shared_params = 0;
  if (layer_start == 0) {
    if (cfg.vocab_size > 0 && cfg.hidden_size > 0) {
      shared_params += static_cast<std::size_t>(cfg.vocab_size) * static_cast<std::size_t>(cfg.hidden_size);  // embedding
      shared_params += static_cast<std::size_t>(cfg.vocab_size) * static_cast<std::size_t>(cfg.hidden_size);  // lm_head (rough)
      shared_params += static_cast<std::size_t>(cfg.hidden_size);  // final norm (rough)
    }
  }

  return (static_cast<std::size_t>(n_layers) * per_layer_params + shared_params) * dtype_bytes;
}

}  // namespace qwen
