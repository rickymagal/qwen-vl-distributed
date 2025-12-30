#pragma once

#include "model/model_stage.h"
#include "runtime/activation_packet.h"

#include <cstdint>
#include <optional>
#include <string>

namespace qwen {

class PipelineStage {
public:
  explicit PipelineStage(const ModelConfig& cfg);

  // Local execution (no transport): takes StageInput, returns StageOutput.
  StageOutput run_local(const StageInput& in);

  // Deserialize an ActivationPacket into StageInput, run_local(), return StageOutput.
  StageOutput run_from_activation(const ActivationPacket& p, int device_index);

  // Serialize StageOutput into ActivationPacket to send to next stage.
  ActivationPacket to_activation(const StageOutput& out,
                                 int32_t stage_from,
                                 int32_t stage_to,
                                 int64_t step,
                                 int64_t pos);

private:
  ModelConfig cfg_;
  ModelStage stage_;
};

} // namespace qwen
