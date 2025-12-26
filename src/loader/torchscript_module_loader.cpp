#include "loader/weight_loader.h"

#include <torch/script.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen {

// This loader treats a TorchScript module as the "artifact" and extracts named parameters/buffers
// into a key->tensor map. This is useful if export_model.py produces model.ts.pt.
class TorchScriptModuleWeightLoader final : public WeightLoader {
public:
  explicit TorchScriptModuleWeightLoader(const std::string& path) {
    module_ = torch::jit::load(path);

    // Parameters
    for (const auto& p : module_.named_parameters(/*recurse=*/true)) {
      tensors_.emplace(p.name, p.value);
    }
    // Buffers
    for (const auto& b : module_.named_buffers(/*recurse=*/true)) {
      tensors_.emplace(b.name, b.value);
    }

    if (tensors_.empty()) {
      throw std::runtime_error("TorchScriptModuleWeightLoader: no params/buffers found in " + path);
    }
  }

  bool exists(const std::string& key) const override {
    return tensors_.find(key) != tensors_.end();
  }

  torch::Tensor get(const std::string& key) const override {
    auto it = tensors_.find(key);
    if (it == tensors_.end()) throw std::runtime_error("TorchScriptModuleWeightLoader: missing key: " + key);
    return it->second;
  }

  std::vector<std::string> list_keys() const override {
    std::vector<std::string> ks;
    ks.reserve(tensors_.size());
    for (const auto& kv : tensors_) ks.push_back(kv.first);
    return ks;
  }

private:
  torch::jit::Module module_;
  std::unordered_map<std::string, torch::Tensor> tensors_;
};

// Factory (declare later in a header if desired)
std::unique_ptr<WeightLoader> make_torchscript_weight_loader(const std::string& path) {
  return std::make_unique<TorchScriptModuleWeightLoader>(path);
}

} // namespace qwen
