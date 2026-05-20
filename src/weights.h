#pragma once

struct LayerWeights {
    float* input_layernorm;
    float* q_proj;
    float* k_proj;
    float* v_proj;
    float* o_proj;
    float* post_attn_layernorm;
    float* gate_proj;
    float* up_proj;
    float* down_proj;

    float* q_bias;
    float* k_bias;
    float* v_bias;
};

struct ModelWeights {
    float* embed_tokens;
    float* final_norm;
    LayerWeights layers[24];
};
