#include "runtime/pipeline_stage.h"

namespace qwen {

PipelineStage::PipelineStage(const ModelConfig& cfg)
    : cfg_(cfg),
      stage_(cfg_) {}

StageOutput PipelineStage::run_local(const StageInput& in) {
  return stage_->forward(in);
}

StageOutput PipelineStage::run_from_activation(const ActivationPacket& p, int device_index) {
  (void)device_index;

  StageInput in;

  // Only set fields we know exist from your compile errors / prior context.
  in.pos = p.pos;
  in.hidden_in = p.hidden;
  in.attn_mask = p.attn_mask;

  return run_local(in);
}

ActivationPacket PipelineStage::to_activation(const StageOutput& out,
                                              int32_t stage_from,
                                              int32_t stage_to,
                                              int64_t step,
                                              int64_t pos) {
  (void)step;

  ActivationPacket p;
  p.version = 1;
  p.stage_from = stage_from;
  p.stage_to = stage_to;
  p.pos = pos;

  p.hidden = out.hidden_out;

  // Do NOT set p.attn_mask / p.logits here: those fields don't exist in your ActivationPacket.

  return p;
}

} // namespace qwen
