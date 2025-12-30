#include "model/model_stage.h"

#include "core/tensor_utils.h"

namespace qwen {

ModelStageImpl::ModelStageImpl(const ModelConfig& cfg) : cfg_(cfg) {
  if (cfg_.vocab_size > 0) {
    embedding_ = register_module("embedding", Embedding(cfg_));
    lm_head_ = register_module("lm_head", torch::nn::Linear(cfg_.hidden_size, cfg_.vocab_size));
  }

  if (blocks_.empty() && cfg_.num_hidden_layers > 0) {
    blocks_.reserve((size_t)cfg_.num_hidden_layers);
    for (int32_t i = 0; i < cfg_.num_hidden_layers; ++i) {
      auto blk = TransformerBlock(cfg_, i);
      blocks_.push_back(register_module("block_" + std::to_string(i), blk));
    }
  }
}

StageOutput ModelStageImpl::forward(const StageInput& in) {
  StageOutput out;

  torch::Tensor h = in.hidden_in;
  if (in.input_ids.defined()) {
    require((bool)embedding_, "ModelStage: embedding not initialized");
    h = embedding_->forward(in.input_ids);
  }

  require(h.defined(), "ModelStage: hidden_in is undefined");
  require_cuda(h, "ModelStage: hidden_in must be CUDA");
  require(h.dim() == 3, "ModelStage: expected hidden_in [B,T,D]");

  KVCache* kv = nullptr;
  std::optional<RopeTables> rope = std::nullopt;

  for (auto& blk : blocks_) {
    h = blk->forward(h, in.attn_mask, kv, in.pos, rope);
  }

  out.hidden_out = h;

  if ((bool)lm_head_) {
    out.logits = lm_head_->forward(h);
  }

  return out;
}

} // namespace qwen
