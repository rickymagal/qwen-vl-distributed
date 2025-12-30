#include "loader/pt_weight_loader.h"

#include <torch/script.h>

#include <stdexcept>
#include <unordered_map>

namespace qwen {

PtWeightLoader::PtWeightLoader(std::string path) : path_(std::move(path)) {}

std::unordered_map<std::string, torch::Tensor> PtWeightLoader::load_all() {
  std::unordered_map<std::string, torch::Tensor> out;

  try {
    torch::jit::Module m = torch::jit::load(path_);
    for (const auto& p : m.named_parameters(/*recurse=*/true)) {
      out.emplace(p.name, p.value);
    }
    for (const auto& b : m.named_buffers(/*recurse=*/true)) {
      out.emplace(b.name, b.value);
    }
    return out;
  } catch (const c10::Error&) {
    throw std::runtime_error(
        "PtWeightLoader: torch::jit::load failed. "
        "LibTorch can load TorchScript modules saved with torch.jit.save(), "
        "but it cannot load a Python-pickled state_dict saved with torch.save(state_dict).");
  }
}

} // namespace qwen
