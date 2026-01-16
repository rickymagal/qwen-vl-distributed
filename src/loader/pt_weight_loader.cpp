#include <loader/pt_weight_loader.h>

#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <caffe2/serialize/inline_container.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <filesystem>
#include <optional>
#include <sstream>

namespace qwen {

PtWeightLoader::PtWeightLoader(std::string weights_path)
    : weights_path_(std::move(weights_path)) {}

bool PtWeightLoader::load() {
  weights_.clear();

  std::string err_ts;
  if (try_load_torchscript_(&err_ts)) {
    return true;
  }

  std::string err_sd;
  if (try_load_packed_state_dict_(&err_sd)) {
    return true;
  }

  std::ostringstream oss;
  oss << "PtWeightLoader: failed to load weights from '" << weights_path_ << "'";
  if (!err_ts.empty()) oss << " [torchscript: " << err_ts << "]";
  if (!err_sd.empty()) oss << " [state_dict: " << err_sd << "]";
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

bool PtWeightLoader::try_load_packed_state_dict_(std::string* err) {
  try {
    caffe2::serialize::PyTorchStreamReader reader(weights_path_);
    const auto records = reader.getAllRecords();
    std::string archive;
    for (const auto& r : records) {
      const std::string suffix = "/data.pkl";
      if (r.size() > suffix.size() && r.compare(r.size() - suffix.size(), suffix.size(), suffix) == 0) {
        archive = r.substr(0, r.size() - suffix.size());
        break;
      }
    }

    auto iv = torch::jit::readArchiveAndTensors(
        archive,
        "data",
        "data/",
        std::nullopt,
        std::nullopt,
        std::nullopt,
        reader);

    if (!iv.isGenericDict()) {
      if (err) *err = "packed data is not a dict";
      return false;
    }

    auto dict = iv.toGenericDict();
    if (dict.empty()) {
      if (err) *err = "packed dict is empty";
      return false;
    }

    for (const auto& item : dict) {
      if (!item.key().isString()) continue;
      if (!item.value().isTensor()) continue;
      weights_[item.key().toStringRef()] = item.value().toTensor().detach().cpu();
    }

    if (weights_.empty()) {
      if (err) *err = "packed dict contained no tensor entries";
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
