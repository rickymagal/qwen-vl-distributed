#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwen {

struct ModelConfig {
  // Model identity
  std::string model_id;
  std::string revision;

  // DType strings: "fp16", "bf16"
  std::string dtype = "bf16";

  // Transformer core
  int32_t vocab_size = 0;
  int32_t hidden_size = 0;
  int32_t num_hidden_layers = 0;
  int32_t num_attention_heads = 0;
  int32_t num_key_value_heads = 0;
  int32_t intermediate_size = 0;

  // MoE
  bool use_moe = false;
  int32_t num_experts = 0;
  int32_t top_k = 0;

  // RoPE
  float rope_theta = 10000.0f;
  int32_t rope_dim = 0;

  // KV cache
  int32_t max_batch = 1;
  int32_t max_seq_len = 4096;

  // Vision (placeholder fields; actual values come from spec lock)
  int32_t vision_hidden_size = 0;
  int32_t vision_num_layers = 0;

  // Pipeline partitioning (block-wise)
  int32_t stage_id = 0;
  int32_t stage_count = 1;
  int32_t layer_start = 0; // inclusive
  int32_t layer_end = 0;   // exclusive

  // Runtime
  int32_t device_index = 0; // CUDA device index
};

inline bool is_valid_stage_range(const ModelConfig& c) {
  if (c.layer_start < 0 || c.layer_end < 0) return false;
  if (c.layer_start > c.layer_end) return false;
  if (c.num_hidden_layers > 0 && c.layer_end > c.num_hidden_layers) return false;
  return true;
}

} // namespace qwen
