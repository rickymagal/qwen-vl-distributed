#include "model/attention.h"
#include "core/tensor_utils.h"

#include <cmath>

namespace qwen {
namespace {

static torch::Tensor make_causal_mask(int64_t B, int64_t T, int device_index) {
  auto opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA, device_index);
  // mask: [1, 1, T, T] where True means "keep"
  auto i = torch::arange(T, opts.dtype(torch::kInt64));
  auto j = torch::arange(T, opts.dtype(torch::kInt64));
  auto ii = i.view({T, 1});
  auto jj = j.view({1, T});
  auto keep = (jj <= ii); // [T, T]
  return keep.to(opts).view({1, 1, T, T});
}

static torch::Tensor repeat_kv_heads(const torch::Tensor& kv, int64_t q_heads) {
  // kv: [B, kv_heads, T, D]
  // If kv_heads == q_heads: return kv
  // Else repeat along head dimension (assumes q_heads % kv_heads == 0)
  if (kv.size(1) == q_heads) return kv;
  const int64_t kv_heads = kv.size(1);
  require(q_heads % kv_heads == 0, "Attention: q_heads must be multiple of kv_heads");
  const int64_t rep = q_heads / kv_heads;
  return kv.repeat({1, rep, 1, 1});
}

} // namespace

AttentionImpl::AttentionImpl(const ModelConfig& cfg, int32_t layer_index_in_stage)
    : cfg_(cfg), layer_index_in_stage_(layer_index_in_stage) {
  require(cfg_.hidden_size > 0, "Attention: cfg.hidden_size must be set");
  require(cfg_.num_attention_heads > 0, "Attention: cfg.num_attention_heads must be set");

  // Projections: D -> D (placeholder; real models may use bias/no-bias variations)
  wq_ = register_module("wq", torch::nn::Linear(cfg_.hidden_size, cfg_.hidden_size));
  wk_ = register_module("wk", torch::nn::Linear(cfg_.hidden_size, cfg_.hidden_size));
  wv_ = register_module("wv", torch::nn::Linear(cfg_.hidden_size, cfg_.hidden_size));
  wo_ = register_module("wo", torch::nn::Linear(cfg_.hidden_size, cfg_.hidden_size));
}

torch::Tensor AttentionImpl::forward(const torch::Tensor& x,
                                     const c10::optional<torch::Tensor>& attn_mask,
                                     KVCache* cache,
                                     int64_t pos,
                                     const c10::optional<RopeTables>& rope) {
  require(x.defined(), "Attention: x is undefined");
  require_cuda(x, "Attention: x");
  require(x.dim() == 3, "Attention: expected x shape [B, T, D]");

  const int64_t B = x.size(0);
  const int64_t T = x.size(1);
  const int64_t D = x.size(2);

  require(D == cfg_.hidden_size, "Attention: hidden_size mismatch");

  const int64_t q_heads = cfg_.num_attention_heads;
  const int64_t kv_heads = (cfg_.num_key_value_heads > 0) ? cfg_.num_key_value_heads : q_heads;
  require(q_heads > 0 && kv_heads > 0, "Attention: heads must be > 0");
  require(D % q_heads == 0, "Attention: hidden_size must be divisible by num_attention_heads");
  const int64_t head_dim = D / q_heads;

  // Project
  auto q = wq_->forward(x); // [B,T,D]
  auto k = wk_->forward(x);
  auto v = wv_->forward(x);

  // Reshape to [B, H, T, Hd]
  q = q.view({B, T, q_heads, head_dim}).transpose(1, 2).contiguous();
  k = k.view({B, T, q_heads, head_dim}).transpose(1, 2).contiguous();
  v = v.view({B, T, q_heads, head_dim}).transpose(1, 2).contiguous();

  // If kv_heads != q_heads, we will interpret k/v as kv_heads by slicing heads first,
  // then repeat to q_heads for compute.
  if (kv_heads != q_heads) {
    require(kv_heads <= q_heads, "Attention: kv_heads > q_heads not supported in scaffold");
    k = k.index({torch::indexing::Slice(), torch::indexing::Slice(0, kv_heads),
                 torch::indexing::Slice(), torch::indexing::Slice()}).contiguous();
    v = v.index({torch::indexing::Slice(), torch::indexing::Slice(0, kv_heads),
                 torch::indexing::Slice(), torch::indexing::Slice()}).contiguous();
  }

  // Apply RoPE (in-place) if provided
  if (rope.has_value() && rope->cos.defined() && rope->sin.defined() && rope->rope_dim > 0) {
    // For k/v we only rope k (standard). Here we rope q and k.
    if (k.size(1) != q_heads) {
      // Temporarily repeat k to q_heads to apply RoPE consistently.
      auto k_rep = repeat_kv_heads(k, q_heads);
      apply_rope_inplace(q, k_rep, *rope, pos);
      // Reduce back to kv_heads by taking first heads before cache append
      k = k_rep.index({torch::indexing::Slice(), torch::indexing::Slice(0, kv_heads),
                       torch::indexing::Slice(), torch::indexing::Slice()}).contiguous();
    } else {
      apply_rope_inplace(q, k, *rope, pos);
    }
  }

  torch::Tensor k_all;
  torch::Tensor v_all;

  if (cache && cache->is_initialized()) {
    // Append new k/v into cache for this layer (cache stores kv_heads layout)
    require(k.dim() == 4 && v.dim() == 4, "Attention: k/v must be [B,kv_heads,T,Hd] for cache");
    cache->append(layer_index_in_stage_, k, v, pos);

    const int64_t S = pos + T;
    auto& lk = cache->layer(layer_index_in_stage_);
    k_all = lk.k.index({torch::indexing::Slice(0, B),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(0, S),
                        torch::indexing::Slice()}).contiguous();
    v_all = lk.v.index({torch::indexing::Slice(0, B),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(0, S),
                        torch::indexing::Slice()}).contiguous();
  } else {
    k_all = k;
    v_all = v;
  }

  // Repeat kv heads to match q heads if needed
  k_all = repeat_kv_heads(k_all, q_heads);
  v_all = repeat_kv_heads(v_all, q_heads);

  const int64_t S = k_all.size(2);

  // Attention scores: [B,H,T,S]
  auto scale = 1.0 / std::sqrt((double)head_dim);
  auto attn_scores = torch::matmul(q, k_all.transpose(-2, -1)) * scale;

  // Masking: support optional provided mask else causal
  if (attn_mask.has_value() && attn_mask->defined()) {
    auto m = *attn_mask;
    require_cuda(m, "Attention: attn_mask");
    // Accept bool mask (keep=true) or additive mask (float)
    if (m.scalar_type() == torch::kBool) {
      // Broadcastable to [B,H,T,S]
      attn_scores = attn_scores.masked_fill(~m, -1e9);
    } else {
      // additive mask: add directly
      attn_scores = attn_scores + m;
    }
  } else {
    // causal within current window (assumes S == T if no cache; else S can be larger)
    // We create a causal mask for the query positions relative to keys.
    // For simplicity, when S > T, allow attending to all past keys:
    // create [1,1,T,S] where key index <= pos + query_index
    auto opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA, x.get_device());
    auto qi = torch::arange(T, opts.dtype(torch::kInt64)).view({T, 1});
    auto kj = torch::arange(S, opts.dtype(torch::kInt64)).view({1, S});
    auto keep = (kj <= (qi + pos)); // [T,S]
    auto mask = keep.to(opts).view({1, 1, T, S});
    attn_scores = attn_scores.masked_fill(~mask, -1e9);
  }

  auto attn_probs = torch::softmax(attn_scores, -1);
  auto ctx = torch::matmul(attn_probs, v_all); // [B,H,T,Hd]

  // Back to [B,T,D]
  auto y = ctx.transpose(1, 2).contiguous().view({B, T, D});
  y = wo_->forward(y);
  return y;
}

} // namespace qwen
