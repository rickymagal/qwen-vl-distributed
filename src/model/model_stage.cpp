#include "model/model_stage.h"
#include "core/tensor_utils.h"

namespace qwen {

ModelStageImpl::ModelStageImpl(const ModelConfig& cfg) : cfg_(cfg) {
  require(is_valid_stage_range(cfg_), "ModelStage: invalid layer range");
  require(cfg_.hidden_size > 0, "ModelStage: cfg.hidden_size must be set");
  require(cfg_.num_attention_heads > 0, "ModelStage: cfg.num_attention_heads must be set");

  // Optional vision path: typically only stage 0 uses it.
  vision_ = register_module("vision", VisionEncoder(cfg_));
  projector_ = register_module("projector", Projector(cfg_));

  // Optional embedding: used when input_ids are provided at this stage.
  if (cfg_.vocab_size > 0) {
    embedding_ = register_module("embedding", Embedding(cfg_));
  }

  const int32_t nblocks = cfg_.layer_end - cfg_.layer_start;
  require(nblocks >= 0, "ModelStage: negative block count");

  blocks_.reserve((size_t)nblocks);
  for (int32_t i = 0; i < nblocks; ++i) {
    blocks_.push_back(register_module("block_" + std::to_string(i),
                                      TransformerBlock(cfg_, i)));
  }

  // KV cache init for this stage if it owns blocks
  if (nblocks > 0) {
    const int32_t kv_heads = (cfg_.num_key_value_heads > 0) ? cfg_.num_key_value_heads : cfg_.num_attention_heads;
    const int32_t head_dim = cfg_.hidden_size / cfg_.num_attention_heads;

    c10::ScalarType dt = (cfg_.dtype == "fp16") ? torch::kFloat16 : torch::kBFloat16;
    cache_.init(nblocks, cfg_.max_batch, cfg_.max_seq_len, kv_heads, head_dim, dt, cfg_.device_index);
  }

  // LM head only on last stage (and only if vocab_size is known)
  if (cfg_.stage_id == cfg_.stage_count - 1 && cfg_.vocab_size > 0) {
    lm_head_ = register_module("lm_head", torch::nn::Linear(cfg_.hidden_size, cfg_.vocab_size));
  }
}

StageOutput ModelStageImpl::forward(const StageInput& in) {
  StageOutput out;

  torch::Tensor hidden;

  // Priority: hidden_in if provided (pipeline activation).
  if (in.hidden_in.defined()) {
    require_cuda(in.hidden_in, "ModelStage: hidden_in");
    require(in.hidden_in.dim() == 3, "ModelStage: hidden_in must be [B,T,D]");
    require(in.hidden_in.size(2) == cfg_.hidden_size, "ModelStage: hidden_in hidden_size mismatch");
    hidden = in.hidden_in;
  } else if (in.input_ids.defined()) {
    require(embedding_ != nullptr, "ModelStage: embedding not initialized (vocab_size missing?)");
    hidden = embedding_->forward(in.input_ids);
  } else {
    // Vision-only stage can start from images and emit hidden tokens.
    hidden = torch::Tensor();
  }

  // Vision path: if images are provided, encode and project.
  torch::Tensor vtok;
  if (in.images.defined()) {
    vtok = vision_->forward(in.images);
    vtok = projector_->forward(vtok);
    require(vtok.dim() == 3, "ModelStage: projected vision tokens must be [B,V,D]");
    require(vtok.size(2) == cfg_.hidden_size, "ModelStage: projector output hidden_size mismatch");
  }

  // Combine vision + text if both exist.
  if (vtok.defined() && hidden.defined()) {
    // Concatenate on sequence dimension: [B, V+T, D]
    hidden = torch::cat({vtok, hidden}, /*dim=*/1);
  } else if (vtok.defined() && !hidden.defined()) {
    hidden = vtok;
  }

  require(hidden.defined(), "ModelStage: no input provided (need hidden_in, input_ids, or images)");
  require_cuda(hidden, "ModelStage: hidden");
  require(hidden.dim() == 3, "ModelStage: hidden must be [B,T,D]");

  // RoPE tables (optional): in real implementation we will precompute once per stage.
  c10::optional<RopeTables> rope_tables = c10::nullopt;
  if (cfg_.rope_dim > 0) {
    c10::ScalarType dt = hidden.scalar_type();
    rope_tables = precompute_cos_sin(cfg_.max_seq_len, cfg_.rope_dim, cfg_.rope_theta, dt, cfg_.device_index);
  }

  // Pass through owned blocks
  for (size_t i = 0; i < blocks_.size(); ++i) {
    hidden = blocks_[i]->forward(hidden, in.attn_mask, (blocks_.empty() ? nullptr : &cache_), in.pos, rope_tables);
  }

  out.hidden_out = hidden;

  // Produce logits only on last stage if lm_head exists
  if (lm_head_ != nullptr) {
    out.logits = lm_head_->forward(hidden);
  } else {
    out.logits = torch::Tensor(); // undefined
  }

  return out;
}

} // namespace qwen
