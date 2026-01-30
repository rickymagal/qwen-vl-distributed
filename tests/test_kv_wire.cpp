#include "mini_test.h"

#include <torch/torch.h>

#include "core/kv_cache.h"
#include "runtime/kv_wire.h"

int main() {
  SKIP_IF(!torch::cuda::is_available(), "CUDA not available");

  qwen::KVCache cache;
  cache.init(/*layers*/2, /*max_batch*/1, /*max_seq*/8, /*kv_heads*/2, /*head_dim*/4, torch::kFloat16, /*device*/0);

  // Fill cache with non-zero values.
  for (int i = 0; i < cache.num_layers(); ++i) {
    cache.layer(i).k.uniform_(0.0, 1.0);
    cache.layer(i).v.uniform_(0.0, 1.0);
  }

  auto packed = qwen::pack_kv_cache(cache);
  CHECK_TRUE(packed.k.defined());
  CHECK_TRUE(packed.v.defined());
  CHECK_EQ(packed.k.dim(), 5);
  CHECK_EQ(packed.v.dim(), 5);
  CHECK_EQ(packed.k.size(0), 2);

  // Restore into a new cache and compare shapes.
  qwen::KVCache cache2;
  cache2.init(/*layers*/2, /*max_batch*/1, /*max_seq*/8, /*kv_heads*/2, /*head_dim*/4, torch::kFloat16, /*device*/0);
  qwen::restore_kv_cache(&cache2, packed.k, packed.v);
  CHECK_TRUE(cache2.layer(0).k.sizes() == cache.layer(0).k.sizes());
  CHECK_TRUE(cache2.layer(1).v.sizes() == cache.layer(1).v.sizes());

  return 0;
}
