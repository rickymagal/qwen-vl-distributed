#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <vector>
#include "core/config.h"
#include "core/kv_cache.h"
#include "core/rope.h"
#include "model/embedding.h"
#include "model/transformer_block.h"
#include "vision/vision_encoder.h"
#include "vision/projector.h"

namespace qwen {

// One pipeline stage of the full model.
// Stage can be:
//  - stage0: vision encoder + projector + optional embedding and first blocks
//  - middle stages: blocks only
//  - last stage: blocks + lm head
//
// For Milestone 1 we define the contract. Implementation will follow in src/model/model_stage.cpp.

struct StageInput {
  torch::Tensor input_ids;     // [B, T] int64 (optional depending on stage role)
  torch::Tensor images;        // vision input (optional)
  torch::Tensor hidden_in;     // [B, T, D] (activation from prev stage)
  int64_t pos = 0;
  c10::optional<torch::Tensor> attn_mask;
};

struct StageOutput {
  torch::Tensor hidden_out;     // [B, T, D]
  torch::Tensor logits;         // [B, T, vocab] optional (only last stage)
};

class ModelStageImpl : public torch::nn::Module {
public:
  explicit ModelStageImpl(const ModelConfig& cfg);

  StageOutput forward(const StageInput& in);

  KVCache& cache() { return cache_; }
  const ModelConfig& cfg() const { return cfg_; }

  // Components (some may be null depending on stage_id / role)
  VisionEncoder& vision() { return vision_; }
  Projector& projector() { return projector_; }
  Embedding& embedding() { return embedding_; }

private:
  ModelConfig cfg_;

  VisionEncoder vision_{nullptr};
  Projector projector_{nullptr};

  Embedding embedding_{nullptr};

  std::vector<TransformerBlock> blocks_;

  torch::nn::Linear lm_head_{nullptr}; // only used on last stage

  KVCache cache_;
};

TORCH_MODULE(ModelStage);

} // namespace qwen
