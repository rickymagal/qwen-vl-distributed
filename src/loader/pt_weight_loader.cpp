#include "loader/weight_loader.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <c10/util/Optional.h>
#include <c10/util/irange.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen {

// Loads a packed state_dict saved from Python with torch.save({key: tensor, ...}, "weights.pt").
// In C++, this expects torch::load(...) to yield an IValue holding a GenericDict[str -> Tensor].
class PtWeightLoader final : public WeightLoader {
public:
  explicit PtWeightLoader(const std::string& path) {
    c10::IValue iv;
    torch::load(iv, path);

    if (!iv.isGenericDict()) {
      throw std::runtime_error("PtWeightLoader: expected a dict in " + path);
    }

    auto d = iv.toGenericDict();
    for (const auto& item : d) {
      if (!item.key().isString()) {
        throw std::runtime_error("PtWeightLoader: non-string key in dict");
      }
      const std::string k = item.key().toStringRef();

      if (!item.value().isTensor()) {
        throw std::runtime_error("PtWeightLoader: non-tensor value for key: " + k);
      }
      tensors_.emplace(k, item.value().toTensor());
    }

    if (tensors_.empty()) {
      throw std::runtime_error("PtWeightLoader: dict is empty: " + path);
    }
  }

  bool exists(const std::string& key) const override {
    return tensors_.find(key) != tensors_.end();
  }

  torch::Tensor get(const std::string& key) const override {
    auto it = tensors_.find(key);
    if (it == tensors_.end()) throw std::runtime_error("PtWeightLoader: missing key: " + key);
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

// Factory (you can declare this in a header later if you want).
std::unique_ptr<WeightLoader> make_pt_weight_loader(const std::string& path) {
  return std::make_unique<PtWeightLoader>(path);
}

} // namespace qwen
