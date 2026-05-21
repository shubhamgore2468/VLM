#include <cuda_runtime.h>

// MATMUL KERNEL

// NAIVE MATMUL KERNEL
// y = x @ w.T + bias
// x: [M, K], w: [N, K] (row-major, weight stored as [out, in]), bias: [N], y: [M, N]

__global__ void matmul_v1(
    const float* x,
    const float* w,
    const float* bias,
    float* y,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int i = 0; i < K; i++) {
        acc += x[row * K + i] * w[col * K + i];
    }
    if (bias) acc += bias[col];
    y[row * N + col] = acc;
}

void matmul_cuda(
    const float* x, const float* w, const float* bias, float* y, int M, int N, int K
) {
    dim3 blockdim(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_v1<<<grid, blockdim>>>(x, w, bias, y, M, N, K);
}
