#include "core/kv_cache.h"
#include "core/tensor_utils.h"

namespace qwen {

void KVCache::init(int32_t num_layers_in_stage,
                   int32_t max_batch,
                   int32_t max_seq_len,
                   int32_t kv_heads,
                   int32_t head_dim,
                   c10::ScalarType dtype,
                   int device_index) {
  require(num_layers_in_stage > 0, "KVCache: num_layers_in_stage must be > 0");
  require(max_batch > 0, "KVCache: max_batch must be > 0");
  require(max_seq_len > 0, "KVCache: max_seq_len must be > 0");
  require(kv_heads > 0, "KVCache: kv_heads must be > 0");
  require(head_dim > 0, "KVCache: head_dim must be > 0");

  num_layers_in_stage_ = num_layers_in_stage;
  max_batch_ = max_batch;
  max_seq_len_ = max_seq_len;
  kv_heads_ = kv_heads;
  head_dim_ = head_dim;
  dtype_ = dtype;
  device_index_ = device_index;

  layers_.clear();
  layers_.resize(num_layers_in_stage_);

  auto opts = torch::TensorOptions().dtype(dtype_).device(torch::kCUDA, device_index_);

  for (int32_t i = 0; i < num_layers_in_stage_; ++i) {
    layers_[i].k = torch::zeros({max_batch_, kv_heads_, max_seq_len_, head_dim_}, opts);
    layers_[i].v = torch::zeros({max_batch_, kv_heads_, max_seq_len_, head_dim_}, opts);
  }

  initialized_ = true;
}

LayerKV& KVCache::layer(int32_t layer_idx) {
  require(initialized_, "KVCache: not initialized");
  require(layer_idx >= 0 && layer_idx < num_layers_in_stage_, "KVCache: layer_idx out of range");
  return layers_[layer_idx];
}

const LayerKV& KVCache::layer(int32_t layer_idx) const {
  require(initialized_, "KVCache: not initialized");
  require(layer_idx >= 0 && layer_idx < num_layers_in_stage_, "KVCache: layer_idx out of range");
  return layers_[layer_idx];
}

void KVCache::clear_all() {
  if (!initialized_) return;
  for (auto& l : layers_) {
    if (l.k.defined()) l.k.zero_();
    if (l.v.defined()) l.v.zero_();
  }
}

void KVCache::append(int32_t layer_idx,
                     const torch::Tensor& new_k,
                     const torch::Tensor& new_v,
                     int64_t pos) {
  require(initialized_, "KVCache: not initialized");
  require(layer_idx >= 0 && layer_idx < num_layers_in_stage_, "KVCache: layer_idx out of range");
  require(pos >= 0, "KVCache: pos must be >= 0");

  require(new_k.defined() && new_v.defined(), "KVCache: new_k/new_v must be defined");
  require(new_k.is_cuda() && new_v.is_cuda(), "KVCache: new_k/new_v must be CUDA");
  require(new_k.scalar_type() == dtype_ && new_v.scalar_type() == dtype_, "KVCache: dtype mismatch");

  require(new_k.dim() == 4 && new_v.dim() == 4, "KVCache: new_k/new_v must be [B, kv_heads, T, head_dim]");
  require(new_k.size(0) <= max_batch_, "KVCache: batch > max_batch");
  require(new_k.size(1) == kv_heads_, "KVCache: kv_heads mismatch");
  require(new_k.size(3) == head_dim_, "KVCache: head_dim mismatch");

  require(new_v.sizes() == new_k.sizes(), "KVCache: new_v shape mismatch vs new_k");

  const int64_t B = new_k.size(0);
  const int64_t T = new_k.size(2);

  require(pos + T <= max_seq_len_, "KVCache: append would exceed max_seq_len");

  auto& l = layers_[layer_idx];

  // Slice destination [0:B, :, pos:pos+T, :]
  auto dst_k = l.k.index({torch::indexing::Slice(0, B),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(pos, pos + T),
                          torch::indexing::Slice()});
  auto dst_v = l.v.index({torch::indexing::Slice(0, B),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(pos, pos + T),
                          torch::indexing::Slice()});

  dst_k.copy_(new_k);
  dst_v.copy_(new_v);
}

} // namespace qwen
