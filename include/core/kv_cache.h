#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <string>

namespace qwen {

// KV cache owner for one pipeline stage.
// Stores per-layer key/value for self-attention.
//
// Layout (common):
//  k: [B, kv_heads, max_seq, head_dim]
//  v: [B, kv_heads, max_seq, head_dim]
//
// Notes:
// - This is a minimal cache container for Milestone 1 scaffolding.
// - Attention implementation will decide exact layout; keep this stable and explicit.

struct LayerKV {
  torch::Tensor k;
  torch::Tensor v;
};

class KVCache {
public:
  KVCache() = default;

  void init(int32_t num_layers_in_stage,
            int32_t max_batch,
            int32_t max_seq_len,
            int32_t kv_heads,
            int32_t head_dim,
            c10::ScalarType dtype,
            int device_index);

  bool is_initialized() const { return initialized_; }

  int32_t num_layers() const { return num_layers_in_stage_; }
  int32_t max_batch() const { return max_batch_; }
  int32_t max_seq_len() const { return max_seq_len_; }
  int32_t kv_heads() const { return kv_heads_; }
  int32_t head_dim() const { return head_dim_; }

  LayerKV& layer(int32_t layer_idx);
  const LayerKV& layer(int32_t layer_idx) const;

  void clear_all();

  // Append K/V at positions [pos, pos+T)
  // new_k/new_v expected: [B, kv_heads, T, head_dim]
  void append(int32_t layer_idx,
              const torch::Tensor& new_k,
              const torch::Tensor& new_v,
              int64_t pos);

private:
  bool initialized_ = false;
  int32_t num_layers_in_stage_ = 0;
  int32_t max_batch_ = 0;
  int32_t max_seq_len_ = 0;
  int32_t kv_heads_ = 0;
  int32_t head_dim_ = 0;
  c10::ScalarType dtype_ = c10::ScalarType::Half;
  int device_index_ = 0;

  std::vector<LayerKV> layers_;
};

} // namespace qwen
