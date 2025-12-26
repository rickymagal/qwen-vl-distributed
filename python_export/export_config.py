from dataclasses import dataclass

@dataclass
class ExportConfig:
    model_id: str = "Qwen/Qwen3-VL-235B-A22B-Thinking"
    revision: str | None = None
    dtype: str = "bfloat16"
    output_dir: str = "exported"
    use_torchscript: bool = True
