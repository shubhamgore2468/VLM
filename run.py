import torch
from torch.utils.cpp_extension import load
from load_weights import load_qwen_weights

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

x = torch.load("/kaggle/input/datasets/foxtrot22/vlm-reference-tensors/merged_embeds_0.pt").to("cuda")
out = engine.forward(x)

print(f"out shape: {out.shape}, matches input: {torch.equal(out, x)}")