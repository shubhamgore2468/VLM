#pragma once

void rmsnorm_cuda(const float* x, const float* weight, float* y, int seq_len, int hidden_size, float eps);

void matmul_cuda(const float* x, const float* W, const float* bias, float* y,int M, int N, int K);