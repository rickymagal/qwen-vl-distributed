#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>

#include <cstdint>

namespace qwen {

struct KVPacket {
  int32_t version = 1;

  int32_t stage_from = 0;
  int32_t stage_to = 0;

  int64_t step = 0;
  int64_t pos = 0;

  // Minimal representation: cache tensors can be packed however your runtime chooses.
  // Keeping them optional allows "no-kv" paths to work.
  c10::optional<torch::Tensor> k;
  c10::optional<torch::Tensor> v;
};

} // namespace qwen
