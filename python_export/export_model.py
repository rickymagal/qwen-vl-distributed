import os
import torch
from transformers import AutoModel, AutoConfig
from safetensors.torch import save_file
from export_config import ExportConfig

def export_model(cfg: ExportConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }[cfg.dtype]

    config = AutoConfig.from_pretrained(
        cfg.model_id,
        revision=cfg.revision,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        cfg.model_id,
        revision=cfg.revision,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="cpu"
    )

    model.eval()

    if cfg.use_torchscript:
        try:
            scripted = torch.jit.script(model)
            out_path = os.path.join(cfg.output_dir, "model.ts.pt")
            scripted.save(out_path)
            print(f"TorchScript model saved to {out_path}")
            return
        except Exception as e:
            print("TorchScript failed, falling back to state_dict export")
            print(e)

    state_dict = model.state_dict()
    out_path = os.path.join(cfg.output_dir, "model.safetensors")
    save_file(state_dict, out_path)
    print(f"State_dict saved to {out_path}")

if __name__ == "__main__":
    cfg = ExportConfig()
    export_model(cfg)
