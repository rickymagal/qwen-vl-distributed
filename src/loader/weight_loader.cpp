#include "loader/weight_loader.h"

// This translation unit is intentionally minimal.
//
// The weight loader interfaces and the MapWeightLoader implementation are
// header-only (see include/loader/weight_loader.h).
//
// Historical note:
// An older draft implementation of PtWeightLoader lived in this file, but the
// authoritative implementation is now in src/loader/pt_weight_loader.cpp.
//
// Keeping this file avoids build-system churn when the project uses a glob like
// src/loader/*.cpp, while ensuring there are no duplicate symbols.
