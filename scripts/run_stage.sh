#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "usage: $0 <stage_binary> [args...]"
  exit 1
fi

STAGE="$1"
shift

echo "[run_stage] running $STAGE"
./"$STAGE" "$@"
