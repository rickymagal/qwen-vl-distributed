#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d ".venv" ]]; then
  echo "[smoke] missing .venv; create it first"
  exit 2
fi

source .venv/bin/activate

python - <<'PY'
import torch
print("[smoke] torch:", torch.__version__)
print("[smoke] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
  print("[smoke] cuda device:", torch.cuda.get_device_name(0))
PY

OUT_DIR="$ROOT/build_smoke"
mkdir -p "$OUT_DIR"

python - <<'PY'
import os
import shlex
import torch
from torch.utils.cpp_extension import include_paths, library_paths

incs = include_paths()
libs = library_paths()

print("INCLUDES=" + " ".join(shlex.quote(i) for i in incs))
print("LIBS=" + " ".join(shlex.quote(l) for l in libs))
PY > "$OUT_DIR/torch_paths.env"

source "$OUT_DIR/torch_paths.env"

CXX="${CXX:-g++}"
EXE="$OUT_DIR/smoke_forward"

$CXX \
  -O2 -std=c++17 \
  -I"$ROOT/include" \
  $INCLUDES \
  -L"$ROOT/build" \
  $LIBS \
  -Wl,-rpath,'$ORIGIN' \
  -Wl,-rpath,"$(python - <<'PY'
import torch
import os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)" \
  "$ROOT/tests/test_smoke_forward.cu" \
  -o "$EXE" \
  -ltorch -ltorch_cuda -lc10 -lc10_cuda

echo "[smoke] running: $EXE"
"$EXE"
