#pragma once

#include <torch/torch.h>

#include <string>
#include <unordered_map>

namespace qwen {

class PtWeightLoader {
 public:
  explicit PtWeightLoader(std::string weights_path);

  bool load();

  const std::unordered_map<std::string, torch::Tensor>& weights() const { return weights_; }

 private:
  bool try_load_torchscript_(std::string* err);
  bool load_from_torchscript_(torch::jit::Module& m, std::string* err);
  bool try_load_packed_state_dict_(std::string* err);

  std::string weights_path_;
  std::unordered_map<std::string, torch::Tensor> weights_;
};

} // namespace qwen
