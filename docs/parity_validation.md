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

## Vision Coverage Validation (no full weights)

Use the manifest-based validator to check that vision keys and block fields are present:

```bash
python_export/validate_vision_manifest.py \
  --hf-config /path/to/hf_config.json \
  --manifest /path/to/weights_manifest.json \
  --min-blocks 1
```

The repo also includes a minimal fixture for unit testing:
`tests/fixtures/vision_manifest.json`.

## Distributed Parity (multi-machine)

Run one process per stage on each machine. Each non-first stage listens on a port
and forwards activations to the next stage host.

Stage 0 (machine A):

```bash
./build/distributed_parity_stage \
  --hf-config /path/to/hf_config.json \
  --weights /path/to/weights.pt \
  --num-stages 4 \
  --stage-idx 0 \
  --next-host <hostB> \
  --next-port 5001 \
  --input-ids /path/to/input_ids.pt
```

Stage 1 (machine B):

```bash
./build/distributed_parity_stage \
  --hf-config /path/to/hf_config.json \
  --weights /path/to/weights.pt \
  --num-stages 4 \
  --stage-idx 1 \
  --listen 5001 \
  --next-host <hostC> \
  --next-port 5002
```

Stage 2 (machine C):

```bash
./build/distributed_parity_stage \
  --hf-config /path/to/hf_config.json \
  --weights /path/to/weights.pt \
  --num-stages 4 \
  --stage-idx 2 \
  --listen 5002 \
  --next-host <hostD> \
  --next-port 5003
```

Stage 3 (machine D, final):

```bash
./build/distributed_parity_stage \
  --hf-config /path/to/hf_config.json \
  --weights /path/to/weights.pt \
  --num-stages 4 \
  --stage-idx 3 \
  --listen 5003 \
  --out /path/to/distributed_out.pt
```

Compare `/path/to/distributed_out.pt` against a Python reference (see above).

## Distributed Parity (single machine, multi-process)

Use the same binary with loopback and different ports. Example for 4 stages:

Stage 0:

```bash
./build/distributed_parity_stage \
  --hf-config /path/to/hf_config.json \
  --weights /path/to/weights.pt \
  --num-stages 4 \
  --stage-idx 0 \
  --next-host 127.0.0.1 \
  --next-port 5001 \
  --input-ids /path/to/input_ids.pt
```

Stage 1:

```bash
./build/distributed_parity_stage \
  --hf-config /path/to/hf_config.json \
  --weights /path/to/weights.pt \
  --num-stages 4 \
  --stage-idx 1 \
  --listen 5001 \
  --next-host 127.0.0.1 \
  --next-port 5002
```

Stage 2:

```bash
./build/distributed_parity_stage \
  --hf-config /path/to/hf_config.json \
  --weights /path/to/weights.pt \
  --num-stages 4 \
  --stage-idx 2 \
  --listen 5002 \
  --next-host 127.0.0.1 \
  --next-port 5003
```

Stage 3:

```bash
./build/distributed_parity_stage \
  --hf-config /path/to/hf_config.json \
  --weights /path/to/weights.pt \
  --num-stages 4 \
  --stage-idx 3 \
  --listen 5003 \
  --out /tmp/distributed_out.pt
```

## C++ vs C++ Parity (distributed vs single)

1) Run `parity_runner` for a single-process output:

```bash
./build/parity_runner \
  --hf-config /path/to/hf_config.json \
  --weights /path/to/weights.pt \
  --out /tmp/single_out.pt \
  --input-ids /path/to/input_ids.pt
```

2) Run the distributed pipeline (multi-process or multi-machine) to produce `/tmp/distributed_out.pt`.

3) Compare outputs:

```bash
python_export/compare_tensors.py \
  --a /tmp/single_out.pt \
  --b /tmp/distributed_out.pt
```

## Distributed Transport Integrity Check

This check validates activation transport over TCP by sending a tensor and verifying a checksum.

Server (receiver):

```bash
./build/distributed_transport_check \
  --mode server \
  --port 6000
```

Client (sender):

```bash
./build/distributed_transport_check \
  --mode client \
  --host <server_host> \
  --port 6000 \
  --shape 1,8,64 \
  --dtype fp16 \
  --seed 1234
```

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
