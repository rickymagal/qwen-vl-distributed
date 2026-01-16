# Parity Validation (Milestone 3)

This document describes how to validate C++ outputs against the Python reference.

## Prereqs

- Export artifacts:
  - `hf_config.json`
  - `weights.pt`
- Built `parity_runner` (`build/parity_runner`)
- Python venv with `torch` + `transformers`

## Run (full model)

```bash
python_export/.venv/bin/python python_export/parity_check.py \
  --model-id Qwen/Qwen3-VL-235B-A22B-Thinking \
  --revision <rev> \
  --hf-config /path/to/hf_config.json \
  --weights /path/to/weights.pt \
  --cpp-bin build/parity_runner \
  --out-dir parity_out \
  --dtype bf16
```

Outputs:
- `parity_out/ref.pt` (Python reference output)
- `parity_out/cpp_out.pt` (C++ output)
- `parity_out/parity_report.json` (max/mean abs error and max rel error)

## Reduced Export (no full weights)

If you cannot download full weights, use synthetic reduced exports:

```bash
python_export/.venv/bin/python python_export/reduced_export.py \
  --model-id Qwen/Qwen3-VL-235B-A22B-Thinking \
  --config-path /tmp/qwen_config.json \
  --out-dir python_export/reduced_export_out \
  --num-layers 2 \
  --override-vocab 256 \
  --override-hidden 64 \
  --override-heads 8 \
  --override-kv-heads 2 \
  --override-intermediate 128 \
  --override-moe-intermediate 128 \
  --override-num-experts 4 \
  --override-top-k 2
```

Then run `parity_runner` directly (C++ only). Python parity is only meaningful with real weights.

### Reduced Export Report (synthetic weights)

Run:

```bash
./build/parity_runner \
  --hf-config python_export/reduced_export_out/hf_config.json \
  --weights python_export/reduced_export_out/weights.pt \
  --out /tmp/reduced_out.pt \
  --report /tmp/reduced_report.json
```

Result:

```json
{
  "loaded": 41,
  "missing": 0,
  "mismatched": 0,
  "extra": 0
}
```

## Tolerances

Default tolerances used by `parity_check.py`:
- `atol=5e-3`
- `rtol=5e-2`

Adjust via flags if the precision differs.
