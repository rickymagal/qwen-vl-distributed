#include "model/model_stage.h"

#include "core/tensor_utils.h"

namespace qwen {

ModelStageImpl::ModelStageImpl(const ModelConfig& cfg) : cfg_(cfg) {
  if (cfg_.vision_hidden_size > 0) {
    vision_ = register_module("vision", VisionEncoder(cfg_));
    projector_ = register_module("projector", Projector(cfg_));
  }

  if (cfg_.vocab_size > 0 && is_first_stage()) {
    embedding_ = register_module("embedding", Embedding(cfg_));
  }

  if (cfg_.vocab_size > 0 && is_last_stage()) {
    final_norm_ = register_module("final_norm", RmsNorm(cfg_.hidden_size, cfg_.rms_norm_eps));
    lm_head_ = register_module(
        "lm_head",
        torch::nn::Linear(torch::nn::LinearOptions(cfg_.hidden_size, cfg_.vocab_size).bias(false)));
  }

  const int32_t full_count = (cfg_.num_hidden_layers > 0) ? cfg_.num_hidden_layers : 0;
  int32_t count = block_count();
  if (count <= 0 && cfg_.layer_start == 0 && cfg_.layer_end == 0) {
    count = full_count;
  }

  if (blocks_.empty() && count > 0) {
    blocks_.reserve((size_t)count);
    for (int32_t i = 0; i < count; ++i) {
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

  if (in.images.defined()) {
    require((bool)vision_, "ModelStage: vision encoder not initialized");
    auto vision_h = vision_->forward(in.images);
    if ((bool)projector_) {
      vision_h = projector_->forward(vision_h);
    }
    if (h.defined()) {
      if (vision_h.scalar_type() != h.scalar_type()) {
        vision_h = vision_h.to(h.scalar_type());
      }
      if (vision_h.device() != h.device()) {
        vision_h = vision_h.to(h.device());
      }
    }
    if (h.defined()) {
      h = torch::cat({vision_h, h}, 1);
    } else {
      h = vision_h;
    }
  }

  require(h.defined(), "ModelStage: hidden_in is undefined");
  require_cuda(h, "ModelStage: hidden_in must be CUDA");
  require(h.dim() == 3, "ModelStage: expected hidden_in [B,T,D]");

  KVCache* kv = nullptr;
  c10::optional<RopeTables> rope = c10::nullopt;

  const int32_t n_blocks = static_cast<int32_t>(blocks_.size());
  if (n_blocks > 0) {
    const int32_t kv_heads = (cfg_.num_key_value_heads > 0) ? cfg_.num_key_value_heads : cfg_.num_attention_heads;
    const int32_t head_dim = cfg_.hidden_size / cfg_.num_attention_heads;
    if (!cache_.is_initialized()) {
      cache_.init(n_blocks,
                  cfg_.max_batch > 0 ? cfg_.max_batch : (int32_t)h.size(0),
                  cfg_.max_seq_len > 0 ? cfg_.max_seq_len : (int32_t)h.size(1),
                  kv_heads,
                  head_dim,
                  h.scalar_type(),
                  h.get_device());
    }
    kv = &cache_;

    if (cfg_.rope_dim > 0) {
      const int64_t rope_len = (cfg_.max_seq_len > 0) ? cfg_.max_seq_len : h.size(1);
      const bool need_rebuild =
          !rope_.has_value() ||
          !rope_->cos.defined() ||
          rope_->cos.get_device() != h.get_device() ||
          rope_->cos.scalar_type() != h.scalar_type() ||
          rope_->cos.size(0) < rope_len;
      if (need_rebuild) {
        rope_ = precompute_cos_sin(rope_len, cfg_.rope_dim, cfg_.rope_theta, h.scalar_type(), h.get_device());
      }
      rope = rope_;
    }
  }

  for (auto& blk : blocks_) {
    h = blk->forward(h, in.attn_mask, kv, in.pos, rope);
  }

  out.hidden_out = h;

  if ((bool)lm_head_) {
    if ((bool)final_norm_) {
      h = final_norm_->forward(h);
    }
    out.logits = lm_head_->forward(h);
  }

  return out;
}

} // namespace qwen
