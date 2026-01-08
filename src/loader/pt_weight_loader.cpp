// src/loader/pt_weight_loader.cpp
#include "loader/pt_weight_loader.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen {

PtWeightLoader::PtWeightLoader(const std::string& weights_path) {
  torch::IValue root;
  try {
    torch::load(root, weights_path);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("PtWeightLoader: torch::load failed: ") + e.what());
  }

  if (!root.isGenericDict()) {
    throw std::runtime_error("PtWeightLoader: weights file is not a dict (expected state_dict)");
  }

  const auto d = root.toGenericDict();
  weights_.reserve((size_t)d.size());

  for (const auto& it : d) {
    if (!it.key().isString()) {
      throw std::runtime_error("PtWeightLoader: dict key is not a string");
    }
    const std::string k = it.key().toStringRef();
    if (!it.value().isTensor()) {
      throw std::runtime_error("PtWeightLoader: dict value for key '" + k + "' is not a tensor");
    }
    weights_.emplace(k, it.value().toTensor());
  }

  if (weights_.empty()) {
    throw std::runtime_error("PtWeightLoader: loaded empty state_dict");
  }
}

bool PtWeightLoader::exists(const std::string& key) const {
  return weights_.find(key) != weights_.end();
}

torch::Tensor PtWeightLoader::get(const std::string& key) const {
  auto it = weights_.find(key);
  if (it == weights_.end()) {
    throw std::runtime_error("PtWeightLoader: missing key: " + key);
  }
  return it->second;
}

std::vector<std::string> PtWeightLoader::list_keys() const {
  std::vector<std::string> ks;
  ks.reserve(weights_.size());
  for (const auto& kv : weights_) ks.push_back(kv.first);
  return ks;
}

} // namespace qwen
