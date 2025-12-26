#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <string>
#include "core/config.h"

namespace qwen {

// Multimodal projector interface.
// Takes vision embeddings and projects them to text hidden size (or an adapter space).
//
// For Milestone 1 we define the contract only; the internal structure will be
// finalized during spec lock and implemented in src/vision/projector.cpp.

class ProjectorImpl : public torch::nn::Module {
public:
  explicit ProjectorImpl(const ModelConfig& cfg);

  // vision_emb: [B, V, Dv] -> projected: [B, V, Dtext] (typical)
  torch::Tensor forward(const torch::Tensor& vision_emb);

  const ModelConfig& cfg() const { return cfg_; }

private:
  ModelConfig cfg_;

  // Placeholder; real projector may be MLP or linear stack.
  torch::nn::Linear proj_{nullptr};
};

TORCH_MODULE(Projector);

} // namespace qwen
