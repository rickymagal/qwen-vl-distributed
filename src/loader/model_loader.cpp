#include "loader/model_loader.h"

#include <algorithm>
#include <sstream>

#include "core/tensor_utils.h"

namespace qwen {
namespace {

static void record_used(LoadReport* rep, const std::string& key) {
  if (rep) rep->used_keys.push_back(key);
}

static bool try_assign_param(const WeightLoader& wl,
                             const std::string& key,
                             torch::Tensor& param,
                             LoadReport* rep,
                             bool required,
                             bool strict) {
  if (!wl.exists(key)) {
    if (required && rep) {
      rep->missing++;
      rep->missing_keys.push_back(key);
    }
    return false;
  }

  torch::Tensor src = wl.get(key);
  record_used(rep, key);

  if (!param.defined()) {
    if (rep) {
      rep->mismatched++;
      rep->mismatch_keys.push_back(key + ": param undefined");
    }
    if (strict) throw std::runtime_error("load: param undefined for " + key);
    return false;
  }

  if (src.scalar_type() != param.scalar_type()) {
    src = src.to(param.scalar_type());
  }
  if (src.device() != param.device()) {
    src = src.to(param.device());
  }
  if (!src.is_contiguous()) {
    src = src.contiguous();
  }
  if (src.sizes() != param.sizes()) {
    if (rep) {
      rep->mismatched++;
      std::ostringstream oss;
      oss << key << ": expected " << qwen::shape_str(param) << " got " << qwen::shape_str(src);
      rep->mismatch_keys.push_back(oss.str());
    }
    if (strict) throw std::runtime_error("load: shape mismatch for " + key);
    return false;
  }

  param.detach().copy_(src);
  if (rep) rep->loaded++;
  return true;
}

static bool try_assign_linear_transpose(const torch::Tensor& src,
                                        torch::Tensor& param,
                                        LoadReport* rep,
                                        const std::string& key,
                                        bool strict) {
  torch::Tensor t = src;
  if (t.scalar_type() != param.scalar_type()) {
    t = t.to(param.scalar_type());
  }
  if (t.device() != param.device()) {
    t = t.to(param.device());
  }
  if (!t.is_contiguous()) {
    t = t.contiguous();
  }

  if (t.sizes() == param.sizes()) {
    param.detach().copy_(t);
    if (rep) rep->loaded++;
    return true;
  }
  if (t.dim() == 2 && t.transpose(0, 1).sizes() == param.sizes()) {
    param.detach().copy_(t.transpose(0, 1).contiguous());
    if (rep) rep->loaded++;
    return true;
  }
  if (rep) {
    rep->mismatched++;
    std::ostringstream oss;
    oss << key << ": expected " << qwen::shape_str(param) << " got " << qwen::shape_str(src);
    rep->mismatch_keys.push_back(oss.str());
  }
  if (strict) throw std::runtime_error("load: shape mismatch for " + key);
  return false;
}

static bool try_load_gate_up_combined(const torch::Tensor& gate_up,
                                      ExpertMLP& ex,
                                      LoadReport* rep,
                                      const std::string& key,
                                      bool strict) {
  auto gate_w = ex->gate_proj->weight;
  auto up_w = ex->up_proj->weight;
  const int64_t out_gate = gate_w.size(0);

  torch::Tensor t = gate_up;
  if (t.dim() == 2 && t.size(0) == 2 * out_gate) {
    auto gate = t.index({torch::indexing::Slice(0, out_gate), torch::indexing::Slice()});
    auto up = t.index({torch::indexing::Slice(out_gate, 2 * out_gate), torch::indexing::Slice()});
    try_assign_linear_transpose(gate, ex->gate_proj->weight, rep, key + ":gate", strict);
    try_assign_linear_transpose(up, ex->up_proj->weight, rep, key + ":up", strict);
    return true;
  }
  if (t.dim() == 2 && t.size(1) == 2 * out_gate) {
    auto gate = t.index({torch::indexing::Slice(), torch::indexing::Slice(0, out_gate)});
    auto up = t.index({torch::indexing::Slice(), torch::indexing::Slice(out_gate, 2 * out_gate)});
    try_assign_linear_transpose(gate, ex->gate_proj->weight, rep, key + ":gate", strict);
    try_assign_linear_transpose(up, ex->up_proj->weight, rep, key + ":up", strict);
    return true;
  }
  return false;
}

} // namespace

bool load_stage_weights(ModelStage& stage,
                        const WeightLoader& wl,
                        const ModelConfig& cfg,
                        LoadReport* rep,
                        const LoadOptions& opts) {
  const std::string lm_prefix = "model.language_model";
  const bool strict = opts.strict;

  if ((bool)stage->embedding()) {
    try_assign_param(wl, lm_prefix + ".embed_tokens.weight", stage->embedding()->weight(), rep, true, strict);
  }

  for (size_t i = 0; i < stage->blocks().size(); ++i) {
    const int32_t layer = cfg.layer_start + static_cast<int32_t>(i);
    const std::string base = lm_prefix + ".layers." + std::to_string(layer);
    auto& blk = stage->blocks()[i];

    try_assign_param(wl, base + ".input_layernorm.weight", blk->ln1()->weight(), rep, true, strict);
    try_assign_param(wl, base + ".post_attention_layernorm.weight", blk->ln2()->weight(), rep, true, strict);

    auto& attn = blk->attn();
    try_assign_param(wl, base + ".self_attn.q_proj.weight", attn->wq(), rep, true, strict);
    try_assign_param(wl, base + ".self_attn.k_proj.weight", attn->wk(), rep, true, strict);
    try_assign_param(wl, base + ".self_attn.v_proj.weight", attn->wv(), rep, true, strict);
    try_assign_param(wl, base + ".self_attn.o_proj.weight", attn->wo(), rep, true, strict);

    if (cfg.use_qk_norm) {
      attn->enable_qk_norm(true);
      try_assign_param(wl, base + ".self_attn.q_norm.weight", attn->q_norm()->weight(), rep, true, strict);
      try_assign_param(wl, base + ".self_attn.k_norm.weight", attn->k_norm()->weight(), rep, true, strict);
    } else {
      if (wl.exists(base + ".self_attn.q_norm.weight")) {
        attn->enable_qk_norm(true);
        try_assign_param(wl, base + ".self_attn.q_norm.weight", attn->q_norm()->weight(), rep, true, strict);
      }
      if (wl.exists(base + ".self_attn.k_norm.weight")) {
        attn->enable_qk_norm(true);
        try_assign_param(wl, base + ".self_attn.k_norm.weight", attn->k_norm()->weight(), rep, true, strict);
      }
    }

    auto& moe = blk->moe();
    if (moe->is_moe_layer()) {
      try_assign_param(wl, base + ".mlp.gate.weight", moe->router_w(), rep, true, strict);

      const std::string gate_up_key = base + ".mlp.experts.gate_up_proj";
      const std::string down_key = base + ".mlp.experts.down_proj";

      if (!wl.exists(gate_up_key) || !wl.exists(down_key)) {
        if (rep) {
          if (!wl.exists(gate_up_key)) { rep->missing++; rep->missing_keys.push_back(gate_up_key); }
          if (!wl.exists(down_key)) { rep->missing++; rep->missing_keys.push_back(down_key); }
        }
        if (strict) throw std::runtime_error("load: missing MoE expert tensors at " + base);
      } else {
        auto gate_up = wl.get(gate_up_key);
        auto down = wl.get(down_key);
        record_used(rep, gate_up_key);
        record_used(rep, down_key);

        const int32_t E = cfg.num_experts;
        if (gate_up.dim() == 3 && gate_up.size(0) == E) {
          for (int32_t e = 0; e < E; ++e) {
            auto& ex = moe->expert(e);
            auto gate_up_e = gate_up.index({e});
            if (!try_load_gate_up_combined(gate_up_e, ex, rep, gate_up_key, strict) && strict) {
              throw std::runtime_error("load: gate_up_proj shape mismatch for expert " + std::to_string(e));
            }
          }
        } else {
          for (int32_t e = 0; e < E; ++e) {
            auto& ex = moe->expert(e);
            if (!try_load_gate_up_combined(gate_up, ex, rep, gate_up_key, strict)) {
              if (strict) throw std::runtime_error("load: gate_up_proj shape mismatch");
            }
          }
        }

        if (down.dim() == 3 && down.size(0) == E) {
          for (int32_t e = 0; e < E; ++e) {
            auto& ex = moe->expert(e);
            auto down_e = down.index({e});
            try_assign_linear_transpose(down_e, ex->down_proj->weight, rep, down_key, strict);
          }
        } else {
          for (int32_t e = 0; e < E; ++e) {
            auto& ex = moe->expert(e);
            try_assign_linear_transpose(down, ex->down_proj->weight, rep, down_key, strict);
          }
        }
      }
    } else {
      auto& ex = moe->expert(0);
      const std::string gate_key = base + ".mlp.gate_proj.weight";
      const std::string up_key = base + ".mlp.up_proj.weight";
      const std::string down_key = base + ".mlp.down_proj.weight";
      try_assign_param(wl, gate_key, ex->gate_proj->weight, rep, true, strict);
      try_assign_param(wl, up_key, ex->up_proj->weight, rep, true, strict);
      try_assign_param(wl, down_key, ex->down_proj->weight, rep, true, strict);
    }
  }

  if ((bool)stage->final_norm()) {
    try_assign_param(wl, lm_prefix + ".norm.weight", stage->final_norm()->weight(), rep, true, strict);
  }
  if ((bool)stage->lm_head()) {
    if (!try_assign_param(wl, "lm_head.weight", stage->lm_head()->weight, rep, false, strict)) {
      try_assign_param(wl, lm_prefix + ".lm_head.weight", stage->lm_head()->weight, rep, true, strict);
    }
  }

  if (opts.load_vision && (bool)stage->vision()) {
    // Placeholder: actual vision mapping requires exact architecture parity.
    if (rep) {
      rep->skipped++;
      rep->skipped_keys.push_back("vision");
    }
  }

  return true;
}

std::vector<std::string> diff_unused_keys(const WeightLoader& wl,
                                          const std::vector<std::string>& used_keys) {
  std::unordered_set<std::string> used(used_keys.begin(), used_keys.end());
  std::vector<std::string> extra;
  for (const auto& k : wl.list_keys()) {
    if (used.find(k) == used.end()) extra.push_back(k);
  }
  std::sort(extra.begin(), extra.end());
  return extra;
}

} // namespace qwen
