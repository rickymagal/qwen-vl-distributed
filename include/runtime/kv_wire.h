#pragma once

#include <torch/torch.h>

#include "core/kv_cache.h"

namespace qwen {

struct PackedKV {
  torch::Tensor k; // [L, B, H, S, D] on CPU
  torch::Tensor v; // [L, B, H, S, D] on CPU
};

PackedKV pack_kv_cache(const KVCache& cache);
void restore_kv_cache(KVCache* cache, const torch::Tensor& k, const torch::Tensor& v);

} // namespace qwen
