import torch
from torch.utils.cpp_extension import load
from load_weights import load_qwen_weights

REF_DIR = "/kaggle/input/datasets/foxtrot22/vlm-reference-tensors"

engine = load(
    name="vlm_engine",
    sources=[
        "src/engine.cpp",
        "src/kernels/rmsnorm.cu",
    ],
    verbose=True,
)

weights = load_qwen_weights()
engine.init_weights(weights)

x = torch.load(f"{REF_DIR}/norm_in_0.pt").cuda().squeeze(0)
expected = torch.load(f"{REF_DIR}/norm_out_0.pt").cuda().squeeze(0)
norm_weight = weights["model.layers.0.input_layernorm.weight"]

actual = engine.rmsnorm(x, norm_weight, 1e-6)

max_diff = (actual - expected).abs().max().item()
print(f"max diff: {max_diff:.2e}")
print(f"allclose (atol=1e-4): {torch.allclose(actual, expected, atol=1e-4)}")
