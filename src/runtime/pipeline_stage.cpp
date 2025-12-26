#include "runtime/pipeline_stage.h"

#include "core/tensor_utils.h"
#include "model/model_stage.h"
#include "runtime/transport.h"

#include <torch/torch.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace qwen {

PipelineStage::PipelineStage(const ModelConfig& cfg)
    : cfg_(cfg), stage_(cfg) {
  torch::Device device(torch::kCUDA, cfg_.device_index);
  stage_->to(device);
}

StageOutput PipelineStage::run_local(const StageInput& in) {
  return stage_->forward(in);
}

StageOutput PipelineStage::run_from_activation(const ActivationPacket& p, int device_index) {
  torch::Device device(torch::kCUDA, device_index);

  StageInput in;
  in.pos = p.pos;
  in.attn_mask = p.attn_mask;

  if (!p.hidden.defined()) {
    throw std::runtime_error("PipelineStage: activation packet missing hidden");
  }

  auto h = p.hidden;
  if (!h.is_cuda()) h = h.to(device);
  if (!h.is_contiguous()) h = h.contiguous();
  in.hidden_in = h;

  return stage_->forward(in);
}

ActivationPacket PipelineStage::to_activation(const StageOutput& out,
                                             int32_t stage_from,
                                             int32_t stage_to,
                                             int64_t step,
                                             int64_t pos) {
  ActivationPacket p;
  p.version = 1;
  p.stage_from = stage_from;
  p.stage_to = stage_to;
  p.step = step;
  p.pos = pos;
  p.hidden = out.hidden_out;
  return p;
}

} // namespace qwen
