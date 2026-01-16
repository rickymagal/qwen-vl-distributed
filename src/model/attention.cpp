// src/model/attention.cpp
#include "model/attention.h"
#include "core/tensor_utils.h"

#include <cmath>

namespace qwen {
namespace {

// Build a causal keep-mask for the no-cache case: [1,1,T,T] where True means keep.
static torch::Tensor make_causal_keep_mask(int64_t T, int device_index) {
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_index);
  auto i = torch::arange(T, opts_i64).view({T, 1});
  auto j = torch::arange(T, opts_i64).view({1, T});
  auto keep = (j <= i); // [T,T] bool
  auto opts_b = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA, device_index);
  return keep.to(opts_b).view({1, 1, T, T});
}

// Repeat kv heads to match q heads.
// kv: [B, kv_heads, S, Hd] -> [B, q_heads, S, Hd]
static torch::Tensor repeat_kv_heads(const torch::Tensor& kv, int64_t q_heads) {
  if (kv.size(1) == q_heads) return kv;
  const int64_t kv_heads = kv.size(1);
  require(kv_heads > 0, "Attention: kv_heads must be > 0");
  require(q_heads % kv_heads == 0, "Attention: q_heads must be multiple of kv_heads");
  const int64_t rep = q_heads / kv_heads;
  return kv.repeat({1, rep, 1, 1});
}

static torch::Tensor to_keep_bool_mask(const torch::Tensor& m, int device_index) {
  require(m.defined(), "Attention: mask is undefined");
  require_cuda(m, "Attention: mask must be CUDA");
  if (m.scalar_type() == torch::kBool) return m;
  // If it's a float additive mask, we don't convert; caller should handle add.
  // This function is only for bool keep-masks.
  require(false, "Attention: expected bool keep-mask");
  (void)device_index;
  return torch::Tensor();
}

} // namespace

AttentionImpl::AttentionImpl(const ModelConfig& cfg, int32_t layer_index_in_stage)
    : cfg_(cfg), layer_index_in_stage_(layer_index_in_stage) {
  require(cfg_.hidden_size > 0, "Attention: cfg.hidden_size must be set");
  require(cfg_.num_attention_heads > 0, "Attention: cfg.num_attention_heads must be set");

  // Projections: Q is D->D, K/V are D->(kv_heads*head_dim).
  const int64_t q_heads = cfg_.num_attention_heads;
  const int64_t kv_heads = (cfg_.num_key_value_heads > 0) ? cfg_.num_key_value_heads : q_heads;
  require(cfg_.hidden_size % q_heads == 0, "Attention: hidden_size must be divisible by num_attention_heads");
  const int64_t head_dim = cfg_.hidden_size / q_heads;
  const int64_t kv_dim = kv_heads * head_dim;

  wq_ = register_module(
      "wq",
      torch::nn::Linear(torch::nn::LinearOptions(cfg_.hidden_size, cfg_.hidden_size).bias(false)));
  wk_ = register_module(
      "wk",
      torch::nn::Linear(torch::nn::LinearOptions(cfg_.hidden_size, kv_dim).bias(false)));
  wv_ = register_module(
      "wv",
      torch::nn::Linear(torch::nn::LinearOptions(cfg_.hidden_size, kv_dim).bias(false)));
  wo_ = register_module(
      "wo",
      torch::nn::Linear(torch::nn::LinearOptions(cfg_.hidden_size, cfg_.hidden_size).bias(false)));

  q_norm_ = register_module("q_norm", RmsNorm(head_dim, cfg_.rms_norm_eps));
  k_norm_ = register_module("k_norm", RmsNorm(head_dim, cfg_.rms_norm_eps));
  use_qk_norm_ = cfg_.use_qk_norm;
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
  require(kv_heads <= q_heads, "Attention: kv_heads must be <= q_heads");
  require(D % q_heads == 0, "Attention: hidden_size must be divisible by num_attention_heads");
  const int64_t head_dim = D / q_heads;

  // Project: [B,T,D]
  auto q = wq_->forward(x);
  auto k = wk_->forward(x);
  auto v = wv_->forward(x);

  // Shape to [B, H, T, Hd]
  q = q.view({B, T, q_heads, head_dim}).transpose(1, 2).contiguous();
  k = k.view({B, T, kv_heads, head_dim}).transpose(1, 2).contiguous();
  v = v.view({B, T, kv_heads, head_dim}).transpose(1, 2).contiguous();

  if (use_qk_norm_) {
    q = q_norm_->forward(q);
    k = k_norm_->forward(k);
  }

  // Apply RoPE to q and k if provided.
  if (rope.has_value() && rope->cos.defined() && rope->sin.defined() && rope->rope_dim > 0) {
    // RoPE helper expects q and k to have same head count; apply on a temporary repeated k if needed.
    if (k.size(1) != q_heads) {
      auto k_rep = repeat_kv_heads(k, q_heads);
      apply_rope_inplace(q, k_rep, *rope, pos);
      k = k_rep.index({torch::indexing::Slice(),
                       torch::indexing::Slice(0, kv_heads),
                       torch::indexing::Slice(),
                       torch::indexing::Slice()}).contiguous();
    } else {
      apply_rope_inplace(q, k, *rope, pos);
    }
  }

  torch::Tensor k_all;
  torch::Tensor v_all;

  // Cache path: store as [B, kv_heads, S, Hd]
  if (cache && cache->is_initialized()) {
    require(pos >= 0, "Attention: pos must be >= 0");
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

  // Expand kv heads to q heads for attention compute.
  k_all = repeat_kv_heads(k_all, q_heads);
  v_all = repeat_kv_heads(v_all, q_heads);

  const int64_t S = k_all.size(2);

  // Scores: [B,H,T,S]
  const double scale = 1.0 / std::sqrt((double)head_dim);
  auto attn_scores = torch::matmul(q, k_all.transpose(-2, -1)) * scale;

  // Masking: bool keep-mask or additive float mask.
  if (attn_mask.has_value() && attn_mask->defined()) {
    auto m = *attn_mask;
    require_cuda(m, "Attention: attn_mask");
    if (m.scalar_type() == torch::kBool) {
      // keep=true; fill where keep=false
      attn_scores = attn_scores.masked_fill(~m, -1e9);
    } else {
      attn_scores = attn_scores + m;
    }
  } else {
    // Causal masking; if S > T (cache), allow attending to all keys <= pos + t
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, x.get_device());
    auto qi = torch::arange(T, opts_i64).view({T, 1});
    auto kj = torch::arange(S, opts_i64).view({1, S});
    auto keep = (kj <= (qi + pos)); // [T,S]
    auto opts_b = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA, x.get_device());
    auto mask = keep.to(opts_b).view({1, 1, T, S});
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
