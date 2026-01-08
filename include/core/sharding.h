// include/core/sharding.h
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/config.h"

namespace qwen {

struct ShardSpec {
  int32_t stage_id = 0;
  int32_t stage_count = 1;
  int32_t layer_start = 0;  // inclusive
  int32_t layer_end = 0;    // exclusive
  int32_t device_index = 0;

  // Rough, config-only estimates (bytes).
  std::size_t est_weight_bytes = 0;
  std::size_t est_kv_bytes_per_token = 0;
};

struct ShardingPlan {
  std::vector<ShardSpec> stages;
};

std::vector<std::pair<int32_t, int32_t>> shard_layers_even(int32_t num_layers, int32_t stage_count);

ShardingPlan make_plan_even_layers(const ModelConfig& base,
                                   int32_t stage_count,
                                   const std::vector<int32_t>& device_indices);

ShardingPlan make_plan_manual(const ModelConfig& base,
                              const std::vector<std::pair<int32_t, int32_t>>& ranges,
                              const std::vector<int32_t>& device_indices);

ModelConfig config_for_stage(const ModelConfig& base, const ShardSpec& s);

// Estimates (no weights required). These are intentionally rough and should be treated
// as planning numbers, not a guarantee.
std::size_t estimate_kv_bytes_per_token(const ModelConfig& cfg, int32_t layer_start, int32_t layer_end);

std::size_t estimate_weight_bytes_dense_only(const ModelConfig& cfg, int32_t layer_start, int32_t layer_end);

// Same as dense-only, but includes a rough MoE estimate if cfg.use_moe is true.
std::size_t estimate_weight_bytes(const ModelConfig& cfg, int32_t layer_start, int32_t layer_end);

}  // namespace qwen
