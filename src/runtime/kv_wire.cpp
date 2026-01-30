#include "runtime/kv_wire.h"

#include "core/tensor_utils.h"

namespace qwen {

PackedKV pack_kv_cache(const KVCache& cache) {
  PackedKV out;
  if (!cache.is_initialized()) return out;

  const int32_t L = cache.num_layers();
  std::vector<torch::Tensor> ks;
  std::vector<torch::Tensor> vs;
  ks.reserve((size_t)L);
  vs.reserve((size_t)L);

  for (int32_t i = 0; i < L; ++i) {
    auto k = cache.layer(i).k;
    auto v = cache.layer(i).v;
    require(k.defined() && v.defined(), "pack_kv_cache: k/v undefined");
    if (k.is_cuda()) k = k.to(torch::kCPU);
    if (v.is_cuda()) v = v.to(torch::kCPU);
    if (!k.is_contiguous()) k = k.contiguous();
    if (!v.is_contiguous()) v = v.contiguous();
    ks.push_back(k);
    vs.push_back(v);
  }

  out.k = torch::stack(ks, 0);
  out.v = torch::stack(vs, 0);
  return out;
}

void restore_kv_cache(KVCache* cache, const torch::Tensor& k, const torch::Tensor& v) {
  require(cache, "restore_kv_cache: cache is null");
  require(cache->is_initialized(), "restore_kv_cache: cache not initialized");
  require(k.defined() && v.defined(), "restore_kv_cache: k/v undefined");
  require(k.dim() == 5 && v.dim() == 5, "restore_kv_cache: expected [L,B,H,S,D]");
  require(k.sizes() == v.sizes(), "restore_kv_cache: k/v shape mismatch");

  const int32_t L = cache->num_layers();
  require(k.size(0) == L, "restore_kv_cache: layer count mismatch");

  for (int32_t i = 0; i < L; ++i) {
    auto k_i = k.index({i});
    auto v_i = v.index({i});
    if (k_i.is_cuda()) k_i = k_i.to(torch::kCPU);
    if (v_i.is_cuda()) v_i = v_i.to(torch::kCPU);

    auto& lk = cache->layer(i);
    auto k_dst = lk.k;
    auto v_dst = lk.v;
    if (k_dst.is_cuda()) {
      k_i = k_i.to(k_dst.device());
      v_i = v_i.to(v_dst.device());
    }
    k_dst.copy_(k_i);
    v_dst.copy_(v_i);
  }
}

} // namespace qwen
