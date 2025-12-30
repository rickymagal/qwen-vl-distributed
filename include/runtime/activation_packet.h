#pragma once

#include <cstdint>
#include <torch/torch.h>
#include <c10/util/Optional.h>

namespace qwen {

struct ActivationPacket {
  int32_t version = 1;

  int32_t stage_from = 0;
  int32_t stage_to = 0;

  int64_t step = 0;
  int64_t pos = 0;

  torch::Tensor hidden;
  c10::optional<torch::Tensor> attn_mask;
};

} // namespace qwen
