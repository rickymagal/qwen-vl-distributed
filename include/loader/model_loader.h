#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "core/config.h"
#include "loader/weight_loader.h"
#include "model/model_stage.h"

namespace qwen {

struct LoadReport {
  int64_t loaded = 0;
  int64_t missing = 0;
  int64_t mismatched = 0;
  int64_t skipped = 0;
  std::vector<std::string> missing_keys;
  std::vector<std::string> mismatch_keys;
  std::vector<std::string> skipped_keys;
  std::vector<std::string> used_keys;
};

struct LoadOptions {
  bool strict = true;
  bool load_vision = false;
};

// Load weights for a single stage using HF-style keys.
// Returns true on success (or throws if options.strict is true).
bool load_stage_weights(ModelStage& stage,
                        const WeightLoader& wl,
                        const ModelConfig& cfg,
                        LoadReport* report,
                        const LoadOptions& opts = {});

// Utility to compute the set of extra keys in the loader not used by mapping.
std::vector<std::string> diff_unused_keys(const WeightLoader& wl,
                                          const std::vector<std::string>& used_keys);

} // namespace qwen
