#pragma once

void rmsnorm_cuda(const float* x, const float* weight, float* y, int seq_len, int hidden_size, float eps);