#include <torch/extension.h>
#include "weights.h"
using namespace std;


static ModelWeights g_weights;

void init_weights(const map<string, torch::Tensor>& w){
    auto ptr = [&](const string& name) -> float* {
        auto it = w.find(name);
        if(it == w.end()) {
            throw runtime_error("Weight not found: " + name);
        }
        return it->second.data_ptr<float>();
    };

    g_weights.embed_tokens = ptr("model.embed_tokens.weight");
    g_weights.final_norm = ptr("model.norm.weight");

    for (int i=0; i<24; i++){
        string p = "model.layers."+to_string(i) + ".";
        g_weights.layers[i].input_layernorm = ptr(p + "input_layernorm.weight");
        g_weights.layers[i].q_proj = ptr(p + "self_attn.q_proj.weight");
        g_weights.layers[i].k_proj = ptr(p + "self_attn.k_proj.weight");
        g_weights.layers[i].v_proj = ptr(p + "self_attn.v_proj.weight");
        g_weights.layers[i].o_proj = ptr(p + "self_attn.o_proj.weight");
        g_weights.layers[i].q_bias = ptr(p + "self_attn.q_proj.bias");
        g_weights.layers[i].k_bias = ptr(p + "self_attn.k_proj.bias");
        g_weights.layers[i].v_bias = ptr(p + "self_attn.v_proj.bias");
        g_weights.layers[i].post_attn_layernorm = ptr(p+"post_attention_layernorm.weight");
        g_weights.layers[i].gate_proj = ptr(p+"mlp.gate_proj.weight");
        g_weights.layers[i].up_proj = ptr(p+"mlp.up_proj.weight");
        g_weights.layers[i].down_proj = ptr(p+"mlp.down_proj.weight");
    }
    printf("Weights initialized successfully.\n");
}

extern void rmsnorm_cuda(const float* x, const float* weight, float* y, int seq_len, int hidden_size, float eps);

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, float eps){
    auto y = torch::empty_like(x);
    int seq_len = x.size(-2);
    int hidden_size = x.size(-1);
    rmsnorm_cuda(x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(), seq_len, hidden_size, eps);
    return y;
}

torch::Tensor forward(torch::Tensor merged_embeds){
    return merged_embeds;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("init_weights", &init_weights, "Initialize weight pointers");
    m.def("rmsnorm", &rmsnorm, "RMSNorm");
    m.def("forward", &forward, "Forward pass");
}
