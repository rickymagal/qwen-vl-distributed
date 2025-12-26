#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen {

inline void require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

inline void require_cuda(const torch::Tensor& t, const std::string& name) {
  require(t.defined(), name + " is undefined");
  require(t.is_cuda(), name + " must be CUDA tensor");
}

inline void require_contiguous(const torch::Tensor& t, const std::string& name) {
  require(t.is_contiguous(), name + " must be contiguous");
}

inline void require_dtype(const torch::Tensor& t, c10::ScalarType dt, const std::string& name) {
  require(t.scalar_type() == dt, name + " has unexpected dtype");
}

inline torch::Tensor to_cuda(const torch::Tensor& t, int device_index) {
  if (!t.defined()) return t;
  if (t.is_cuda()) return t;
  return t.to(torch::Device(torch::kCUDA, device_index), /*non_blocking=*/false);
}

inline torch::Tensor empty_like_on(const torch::Tensor& ref,
                                   const std::vector<int64_t>& sizes,
                                   c10::ScalarType dtype,
                                   int device_index) {
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_index);
  if (ref.defined()) {
    opts = opts.dtype(dtype).device(torch::kCUDA, device_index);
  }
  return torch::empty(sizes, opts);
}

inline int64_t checked_dim(const torch::Tensor& t, int idx, const std::string& name) {
  require(t.defined(), name + " is undefined");
  require(idx >= 0 && idx < t.dim(), name + " dim index out of range");
  return t.sizes().at(idx);
}

inline std::string shape_str(const torch::Tensor& t) {
  if (!t.defined()) return "<undefined>";
  std::string s = "[";
  for (int i = 0; i < t.dim(); ++i) {
    s += std::to_string(t.size(i));
    if (i + 1 < t.dim()) s += ", ";
  }
  s += "]";
  return s;
}

inline void require_shape(const torch::Tensor& t,
                          const std::vector<int64_t>& expected,
                          const std::string& name) {
  require(t.defined(), name + " is undefined");
  require((int)expected.size() == t.dim(), name + " dim mismatch: got " + std::to_string(t.dim()));
  for (int i = 0; i < (int)expected.size(); ++i) {
    if (expected[i] >= 0) {
      require(t.size(i) == expected[i],
              name + " shape mismatch at dim " + std::to_string(i) +
                  ": got " + std::to_string(t.size(i)) +
                  ", expected " + std::to_string(expected[i]));
    }
  }
}

} // namespace qwen
