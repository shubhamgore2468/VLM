#include <cuda_runtime.h>

// RMSNorm kernel: y = x * rsqrt(mean(x*2)+eps)*weight
// x: [batch_size * seq_len, hidden_size]
// weight: [hidden_size]
// y: [batch_size * seq_len, hidden_size]

__global__ void rmsnorm_kernel(
    const float* x, //starting idx of the row
    const float* weight,
    float* y,
    int hidden_size,
    float eps
) {
    
    int token_idx = blockIdx.x; // Each block processes one token (one row of the input matrix)
    int tid = threadIdx.x; // Each thread processes a portion of the hidden size for the token
    int block_size = blockDim.x; // Number of threads per block

    const float* x_row = x + token_idx * hidden_size; // 0(first) + 4(blockDim)
    float* y_row = y + token_idx * hidden_size;

    // Step 1: Compute the sum of squares for the token
    float local_sum2 = 0.0f;
    for (int i = tid; i<hidden_size; i+=blockDim.x){
        float v = x_row[i];
        local_sum2 += v * v;
    }

    // warp identifiers
    int warp_id = tid / 32; // Assuming 32 threads per warp
    int lane_id = tid % 32;

    // Shared memory holds one value per warp
    __shared__ float warp_sums[32]; // one slot per warp, max 32 warps (1024 threads)

    unsigned mask = 0xFFFFFFFFu; // All threads active
    for(int offset = 16; offset > 0; offset /= 2){
        local_sum2 += __shfl_down_sync(mask, local_sum2, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum2; // Each warp writes its sum to shared memory
    }
    __syncthreads();

    // Step 2: Reduce the warp sums to get the total sum of squares

    if (warp_id == 0) {
        float local_sum = (lane_id < block_size / 32) ? warp_sums[lane_id] : 0.0f; // First warp loads the warp sums

        for( int offset = 16; offset > 0; offset /= 2){
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }

        if ( lane_id == 0){
            warp_sums[0] = local_sum;
        }
    }
    __syncthreads();

    float norm = rsqrtf(warp_sums[0] / hidden_size + eps); // Compute the normalization factor
    for(int i=tid; i<hidden_size; i+=block_size){
        y_row[i] = x_row[i] * norm * weight[i];
    }
}


__global__ void rmsnorm_kernel_v1(
    const float* x,
    const float* weight,
    float* y,
    int hidden_size,
    float eps
) {
    /** BLOCK LEVEL REDUCTION
     * This kernel computes the RMSNorm for each token in the input sequence. 
     * Each block processes one token (i.e., one row of the input matrix).
     * The kernel first computes the sum of squares of the input features for the token,
     * then performs a block-wide reduction to get the total sum of squares, and finally
     * normalizes the input features and applies the weight to produce the output.
     * The shared memory is used to store intermediate sums for the reduction step.
     */
    extern __shared__ float s_data[];
    
    float s_sum2 = 0.0f;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* x_row = x + row * hidden_size;
    float* y_row = y + row * hidden_size;

    // Each thread computes the sum of squares for a portion of the hidden size
    for (int i=tid; i<hidden_size; i+=blockDim.x){
        float v = x_row[i];
        s_sum2 += v * v;
    }
    s_data[tid] = s_sum2;
    __syncthreads();

    // Block-wide reduction in shared memory to compute the total sum of squares
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    float norm = rsqrtf(s_data[0] / hidden_size + eps);

    for(int i=tid; i<hidden_size; i+=blockDim.x){
        y_row[i] = x_row[i] * norm * weight[i];
    }
}

void rmsnorm_cuda(
    const float* x,
    const float* weight,
    float* y,
    int seq_len,
    int hidden_size,
    float eps
) {
    int block_size = 256;
    dim3 grid(seq_len);
    dim3 block(block_size);
    // rmsnorm_kernel_v1<<<grid, block, block_size * sizeof(float)>>>(x, weight, y, hidden_size, eps);
    rmsnorm_kernel<<<grid, block>>>(x, weight, y, hidden_size, eps);
}