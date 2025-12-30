#pragma once

#include <torch/torch.h>

#include <string>
#include <unordered_map>

namespace qwen {

class PtWeightLoader {
public:
  explicit PtWeightLoader(std::string path);

  std::unordered_map<std::string, torch::Tensor> load_all();

private:
  std::string path_;
};

} // namespace qwen
