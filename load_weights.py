from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import torch
from pathlib import Path

# --- config ---
QWEN_REPO   = "Qwen/Qwen2.5-0.5B"
QWEN_DIR    = Path("/kaggle/working/qwen-weights")
QWEN_WEIGHTS = QWEN_DIR / "model.safetensors"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.float32

# --- download ---
snapshot_download(
    repo_id=QWEN_REPO,
    local_dir=QWEN_DIR,
    allow_patterns=["*.safetensors", "*.json"],
)

def load_qwen_weights(path=QWEN_WEIGHTS, device=DEVICE, dtype=DTYPE):
    weights = load_file(path)
    return {k: v.to(device=device, dtype=dtype).contiguous() for k, v in weights.items()}

if __name__ == "__main__":
    w = load_qwen_weights()
    print(f"Loaded {len(w)} tensors")
    for k in list(w.keys())[:5]:
        print(f"  {k}: {w[k].shape} {w[k].dtype}")