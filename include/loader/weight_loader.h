#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>

namespace qwen {

// WeightLoader is a runtime-agnostic interface: it loads tensors by key
// and assigns them into LibTorch parameters/buffers.
//
// For Milestone 1, this is the contract. Actual implementations can include:
//  - TorchScript archive reader (preferred if scriptable)
//  - Packed state_dict reader (.pt)
//  - Safetensors reader (fallback / direct-from-HF)
//
// The loader provides:
//  - exists(key)
//  - get(key) returns a tensor (CPU or CUDA depending on implementation)
//  - list_keys() for mapping validation

class WeightLoader {
public:
  virtual ~WeightLoader() = default;

  virtual bool exists(const std::string& key) const = 0;
  virtual torch::Tensor get(const std::string& key) const = 0;
  virtual std::vector<std::string> list_keys() const = 0;
};

// Simple in-memory loader used for tests or for adapters that pre-load tensors.
class MapWeightLoader final : public WeightLoader {
public:
  MapWeightLoader() = default;

  void insert(const std::string& key, const torch::Tensor& t) {
    tensors_[key] = t;
  }

  bool exists(const std::string& key) const override {
    return tensors_.find(key) != tensors_.end();
  }

  torch::Tensor get(const std::string& key) const override {
    auto it = tensors_.find(key);
    if (it == tensors_.end()) {
      throw std::runtime_error("WeightLoader: missing key: " + key);
    }
    return it->second;
  }

  std::vector<std::string> list_keys() const override {
    std::vector<std::string> ks;
    ks.reserve(tensors_.size());
    for (const auto& kv : tensors_) ks.push_back(kv.first);
    return ks;
  }

private:
  std::unordered_map<std::string, torch::Tensor> tensors_;
};

// Utility to assign tensor into a parameter safely.
inline void assign_param(torch::Tensor& param, const torch::Tensor& value) {
  if (!param.defined()) {
    throw std::runtime_error("assign_param: target param is undefined");
  }
  if (!value.defined()) {
    throw std::runtime_error("assign_param: value is undefined");
  }
  if (param.sizes() != value.sizes()) {
    throw std::runtime_error("assign_param: shape mismatch");
  }
  param.detach().copy_(value);
}

} // namespace qwen
