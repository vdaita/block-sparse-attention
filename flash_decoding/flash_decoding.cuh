#include <cuda_runtime.h>
// #include <torch/extension.h>
#include <stdio.h>

#define D 32
#define BLOCK_WIDTH 32 // perform some type of calculation
#define BLOCK_TOKENS 1

__device__ float warp_max_and_broadcast(float val) {
    // Use warp shuffle reduction to find the maximum value
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    // Broadcast the maximum value to all threads in the warp
    val = __shfl_sync(0xFFFFFFFF, val, 0);
    return val;
}

__device__ float warp_add_and_broadcast(float val) {
    // Use warp shuffle reduction to compute the sum
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    // Broadcast the sum value to all threads in the warp
    val = __shfl_sync(0xFFFFFFFF, val, 0);
    return val;
}


__global__
void shared_split_k_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    const int B,
    const int T
) {
    int tx = threadIdx.x; // represents the hidden dim dimension
    int ty = threadIdx.y; // represents the key index dimension
    int by = blockIdx.y; // represents the key index dimension, but which current block
    int bx = blockIdx.x;
    int num_blocks_for_head = gridDim.y;

    int batch = blockIdx.z;

    int start_token_kv = by * BLOCK_TOKENS + tx;

    // load data into shared memory
    __shared__ float shared_q[D];
    for(int i = 0; i < D; i++){
        shared_q[i] = Q[batch * D + i];
    }

    __syncthreads();

    __shared__ float shared_out[BLOCK_TOKENS][D];
    __shared__ float shared_max[BLOCK_TOKENS];
    __shared__ float shared_sum[BLOCK_TOKENS];

    float max_acc = -INFINITY;
    float sum_acc = 0;

    for(int i = start_token_kv; i < T; i += BLOCK_TOKENS * num_blocks_for_head){
        float acc = 0.0f;
        for(int d = 0; d < D; d++){
            acc += shared_q[d] * K[batch * D * T + i * D + d];
        }

        float old_max_acc = max_acc;
        max_acc = fmaxf(max_acc, acc);
        float alpha = expf(old_max_acc - max_acc);
        float norm_weight = expf(acc - max_acc);

        for(int d = 0; d < D; d++){
            shared_out[tx][d] = shared_out[tx][d] * alpha + norm_weight * V[batch * D * T + i * D + d];
        }
        shared_max[tx] = max_acc;
        shared_sum[tx] = shared_sum[tx] * alpha + norm_weight;
    }

    if(tx == 0){
        for(int i = 1; i < BLOCK_TOKENS; i++){
            float old_max_acc = max_acc;
            max_acc = fmaxf(max_acc, shared_max[i]);
            float alpha = expf(old_max_acc - max_acc);
            float beta = expf(shared_max[i] - max_acc);
            for(int d = 0; d < D; d++){
                shared_out[tx][d] = alpha * shared_out[tx][d] + beta * shared_out[i][d];
            }
            shared_sum[tx] = alpha * shared_sum[tx] + beta * shared_sum[i]; // this is the key learning from this representation
        }

        for(int i = 0; i < D; i++){
            output[batch * num_blocks_for_head * D + bx * D + i] = shared_sum[tx][i];
        }
    }
}

// torch::Tensor forward(
//     torch::Tensor query,
//     torch::Tensor keys,
//     torch::Tensor values
// ){
//     int B = query.size(0);
//     int T = keys.size(1);

//     int num_blocks_for_head = min((T + BLOCK_TOKENS - 1) / BLOCK_TOKENS, 8);
//     dim3 gridDim(1, num_blocks_for_head, B);
//     dim3 blockDim(BLOCK_WIDTH, BLOCK_TOKENS, 1);

//     float* query_ptr = query.data_ptr<float>();
//     float* keys_ptr = keys.data_ptr<float>();
//     float* values_ptr = values.data_ptr<float>();
//     auto output = torch::zeros({B, num_blocks_for_head, D});
//     float* output_ptr = output.data_ptr<float>();

//     shared_split_k_kernel<<<gridDim, blockDim>>>(query_ptr, 
//         keys_ptr, 
//         values_ptr, 
//         output_ptr,
//         B,
//         T
//     );
    
//     return output;
// }