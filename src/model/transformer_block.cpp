#include "model/transformer_block.h"
#include "core/tensor_utils.h"

namespace qwen {

TransformerBlockImpl::TransformerBlockImpl(const ModelConfig& cfg, int32_t layer_index_in_stage)
    : cfg_(cfg), layer_index_in_stage_(layer_index_in_stage) {
  require(cfg_.hidden_size > 0, "TransformerBlock: cfg.hidden_size must be set");

  ln1_ = register_module("ln1", RmsNorm(cfg_.hidden_size, cfg_.rms_norm_eps));
  ln2_ = register_module("ln2", RmsNorm(cfg_.hidden_size, cfg_.rms_norm_eps));

  attn_ = register_module("attn", Attention(cfg_, layer_index_in_stage_));
  moe_  = register_module("moe", Moe(cfg_, layer_index_in_stage_));
}

torch::Tensor TransformerBlockImpl::forward(const torch::Tensor& x,
                                            const c10::optional<torch::Tensor>& attn_mask,
                                            KVCache* cache,
                                            int64_t pos,
                                            const c10::optional<RopeTables>& rope) {
  require(x.defined(), "TransformerBlock: x is undefined");
  require_cuda(x, "TransformerBlock: x");
  require(x.dim() == 3, "TransformerBlock: expected [B,T,D]");

  auto h = ln1_->forward(x);
  auto a = attn_->forward(h, attn_mask, cache, pos, rope);
  auto x1 = x + a;

  auto h2 = ln2_->forward(x1);
  auto m = moe_->forward(h2);
  auto x2 = x1 + m.y;

  return x2;
}

} // namespace qwen
