#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include "core/config.h"
#include "core/kv_cache.h"
#include "core/rope.h"
#include "model/attention.h"
#include "model/moe.h"

namespace qwen {

// Transformer block contract (pre-norm style, residual).
// Implementation will be in src/model/transformer_block.cpp during Milestone 2.

class TransformerBlockImpl : public torch::nn::Module {
public:
  TransformerBlockImpl(const ModelConfig& cfg, int32_t layer_index_in_stage);

  // x: [B, T, D]
  torch::Tensor forward(const torch::Tensor& x,
                        const c10::optional<torch::Tensor>& attn_mask,
                        KVCache* cache,
                        int64_t pos,
                        const c10::optional<RopeTables>& rope);

  const ModelConfig& cfg() const { return cfg_; }

private:
  ModelConfig cfg_;
  int32_t layer_index_in_stage_ = 0;

  torch::nn::LayerNorm ln1_{nullptr};
  torch::nn::LayerNorm ln2_{nullptr};

  Attention attn_{nullptr};
  Moe moe_{nullptr};
};

TORCH_MODULE(TransformerBlock);

} // namespace qwen
