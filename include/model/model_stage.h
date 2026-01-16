// include/model/model_stage.h
#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <vector>
#include <string>

#include "core/config.h"
#include "core/kv_cache.h"
#include "core/rope.h"
#include "model/embedding.h"
#include "model/transformer_block.h"
#include "model/rms_norm.h"
#include "vision/vision_encoder.h"
#include "vision/projector.h"

namespace qwen {

// One pipeline stage of the full model.
//
// Stage can be:
//  - stage0: vision encoder + projector + optional embedding and first blocks
//  - middle stages: blocks only
//  - last stage: blocks + lm head
//
// Milestone 2 focuses on a correct CUDA execution path with a real module graph
// (even if weights are not yet mapped).

struct StageInput {
  torch::Tensor input_ids;     // [B, T] int64 (optional)
  torch::Tensor images;        // [B, C, H, W] CUDA (optional)
  torch::Tensor hidden_in;     // [B, T, D] CUDA (optional)
  int64_t pos = 0;             // starting position for KV cache
  c10::optional<torch::Tensor> attn_mask; // optional attention mask
};

struct StageOutput {
  torch::Tensor hidden_out;    // [B, T, D] CUDA
  torch::Tensor logits;        // [B, T, vocab] CUDA (defined only on last stage)
};

class ModelStageImpl : public torch::nn::Module {
public:
  explicit ModelStageImpl(const ModelConfig& cfg);

  StageOutput forward(const StageInput& in);

  KVCache& cache() { return cache_; }
  const ModelConfig& cfg() const { return cfg_; }

  VisionEncoder& vision() { return vision_; }
  Projector& projector() { return projector_; }
  Embedding& embedding() { return embedding_; }
  RmsNorm& final_norm() { return final_norm_; }
  torch::nn::Linear& lm_head() { return lm_head_; }
  std::vector<TransformerBlock>& blocks() { return blocks_; }

private:
  ModelConfig cfg_;

  VisionEncoder vision_{nullptr};
  Projector projector_{nullptr};
  Embedding embedding_{nullptr};
  RmsNorm final_norm_{nullptr};

  std::vector<TransformerBlock> blocks_;

  torch::nn::Linear lm_head_{nullptr}; // only used on last stage

  KVCache cache_;
  c10::optional<RopeTables> rope_;

private:
  int32_t block_count() const { return cfg_.layer_end - cfg_.layer_start; }
  bool is_first_stage() const { return (cfg_.stage_id == 0); }
  bool is_last_stage() const { return (cfg_.stage_id >= 0) && (cfg_.stage_count > 0) && (cfg_.stage_id == cfg_.stage_count - 1); }
};

TORCH_MODULE(ModelStage);

} // namespace qwen
