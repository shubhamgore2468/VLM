import torch
from torch.utils.cpp_extension import load
from load_weights import load_qwen_weights

engine = load(
    name="vlm_engine",
    sources=["src/engine.cpp", "src/kernels/rmsnorm.cu", "src/kernels/matmul.cu"],
    verbose=True,
)

# Test 1: synthetic
torch.manual_seed(0)
x = torch.randn(7, 128, device="cuda")
W = torch.randn(64, 128, device="cuda")
b = torch.randn(64, device="cuda")

actual = engine.matmul(x, W, b)
expected = x @ W.T + b
print(f"synthetic no-bias: max diff {(engine.matmul(x, W, None) - x @ W.T).abs().max():.2e}")
print(f"synthetic w/ bias: max diff {(actual - expected).abs().max():.2e}")
print(f"allclose: {torch.allclose(actual, expected, atol=1e-3)}")

# Test 2: real Qwen q_proj on layer 0 normed input
weights = load_qwen_weights()
norm_in = torch.load("norm_in_0.pt").cuda().squeeze(0)
norm_w = weights["model.layers.0.input_layernorm.weight"]
normed = engine.rmsnorm(norm_in, norm_w, 1e-6)

q_w = weights["model.layers.0.self_attn.q_proj.weight"]
q_b = weights["model.layers.0.self_attn.q_proj.bias"]

actual_q = engine.matmul(normed, q_w, q_b)
expected_q = torch.nn.functional.linear(normed, q_w, q_b)
print(f"real q_proj: max diff {(actual_q - expected_q).abs().max():.2e}")
print(f"allclose: {torch.allclose(actual_q, expected_q, atol=1e-3)}")