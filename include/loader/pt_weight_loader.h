// include/loader/pt_weight_loader.h
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "loader/weight_loader.h"

namespace qwen {

// Loads a PyTorch-exported weights file created by Python:
//
//   torch.save(model.state_dict(), "weights.pt")
//
// This uses libtorch's torch::load(IValue, path) support to deserialize the
// tensor dictionary without Python in the runtime.
//
// Note: This is an in-memory loader and is intended for development and
// correctness-first bring-up. For huge models, we will need sharded/streaming
// formats (later milestones).
class PtWeightLoader final : public WeightLoader {
 public:
  explicit PtWeightLoader(const std::string& weights_path);

  bool exists(const std::string& key) const override;
  torch::Tensor get(const std::string& key) const override;
  std::vector<std::string> list_keys() const override;

 private:
  std::unordered_map<std::string, torch::Tensor> weights_;
};

} // namespace qwen
