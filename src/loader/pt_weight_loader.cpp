#include <loader/pt_weight_loader.h>

#include <torch/script.h>

#include <c10/util/Exception.h>

#include <sstream>

namespace qwen {

PtWeightLoader::PtWeightLoader(std::string weights_path)
    : weights_path_(std::move(weights_path)) {}

bool PtWeightLoader::load() {
  weights_.clear();

  std::string err;
  if (try_load_torchscript_(&err)) {
    return true;
  }

  if (!err.empty()) {
    std::ostringstream oss;
    oss << "PtWeightLoader: failed to load weights from '" << weights_path_ << "': " << err;
    throw std::runtime_error(oss.str());
  }

  std::ostringstream oss;
  oss << "PtWeightLoader: failed to load weights from '" << weights_path_ << "'";
  throw std::runtime_error(oss.str());
}

bool PtWeightLoader::try_load_torchscript_(std::string* err) {
  try {
    torch::jit::Module m = torch::jit::load(weights_path_);
    return load_from_torchscript_(m, err);
  } catch (const c10::Error& e) {
    if (err) *err = e.what_without_backtrace();
    return false;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

bool PtWeightLoader::load_from_torchscript_(torch::jit::Module& m, std::string* err) {
  try {
    for (const auto& p : m.named_parameters(/*recurse=*/true)) {
      weights_[p.name] = p.value.detach().cpu();
    }
    for (const auto& b : m.named_buffers(/*recurse=*/true)) {
      weights_[b.name] = b.value.detach().cpu();
    }

    if (weights_.empty()) {
      if (err) *err = "TorchScript module contained no parameters or buffers";
      return false;
    }
    return true;
  } catch (const c10::Error& e) {
    if (err) *err = e.what_without_backtrace();
    return false;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

} // namespace qwen
