#include <cuda_runtime.h>
// #include <torch/extension.h>
#include <stdio.h>

#define D 128
#define BLOCK_WIDTH 32 // perform some type of calculation
#define BLOCK_TOKENS 32

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
    int num_blocks_for_head = gridDim.y;

    int batch = blockIdx.z;

    int start_token_kv = by * BLOCK_TOKENS + ty;

    // load data into shared memory
    __shared__ float shared_q[D];
    for(int i = ty * BLOCK_WIDTH + tx; i < D; i += BLOCK_WIDTH * BLOCK_TOKENS){
        shared_q[i] = Q[batch * D + i];
    }

    __syncthreads();

    __shared__ float shared_out[BLOCK_TOKENS][D];
    __shared__ float shared_max[BLOCK_TOKENS];
    __shared__ float shared_sum[BLOCK_TOKENS];
    __shared__ float coalesced_out[D];

    // iterate through a given number of tokens
    float sum_qk = 0.0f;
    float max_qk = -INFINITY;
    float values[D / BLOCK_WIDTH] = {0};
    for(int i = start_token_kv; i < T; i += BLOCK_TOKENS * num_blocks_for_head){ // this means that T must be padded to the nearest multiple of 32
        // load the right token for this dimension
        float acc = 0.0f;
        for(int d = tx; d < D; d += BLOCK_WIDTH){
            acc += shared_q[d] * K[batch * T * D + i * D + d]; // d is related to tx, so memory accesses should be coalesced
        }
        acc = warp_add_and_broadcast(acc); // TODO: implement

        // TODO: make sure you scale the number down and update the local sum
        float prev_max_qk = max_qk;
        max_qk = fmaxf(acc, max_qk);
        float alpha = expf(prev_max_qk - max_qk);
        float normalized_weight = expf(acc - max_qk);
        sum_qk *= alpha;

        // now that the accumulator has the weight for the entire thing
        for(int di = 0; di < 4; di++){
            int d = tx + di * BLOCK_WIDTH;
            values[di] *= alpha;
            values[di] += normalized_weight * V[batch * T * D + i * D + d];
        }
    }
    for(int i = 0; i < D / BLOCK_WIDTH; i++){
        shared_out[ty][tx + i * BLOCK_WIDTH] = values[i];
    }

    printf("%d %d %d max: %f", by, ty, tx, max_qk);
    printf("%d %d %d sum: %f", by, ty, tx, sum_qk);

    if(ty == 0){
        shared_max[ty] = max_qk;
        shared_sum[ty] = sum_qk;
    }
    __syncthreads();
    // **reduce across shared out**

    // right now, each thread in a given warp should have the same sum_qk and max_qk, having just written it out to shared memory
    // however, we need to transpose the grid in theory to load in the right 32 values
   
    // save previous max_qk
    max_qk = shared_max[tx];
    sum_qk = shared_sum[tx];

    // adjust the sum and the values
    float local_max_qk = max_qk;
    max_qk = warp_max_and_broadcast(max_qk);
    float alpha = expf(local_max_qk - max_qk); // expf curr_max - new_max
    sum_qk *= alpha;
    sum_qk = warp_add_and_broadcast(sum_qk); // now, each warp has the same sum from adding up all of the sums adjusted by alpha

    float tm_coalesced_out[D / BLOCK_WIDTH];

    // each warp handles 32 separate tokens for a given dimension
    // meaning each thread must handle the 4 dimensions
    for(int d = ty; d < D; d += BLOCK_WIDTH){
        float value = shared_out[tx][d] * alpha;
        // add the values together
        value = warp_add_and_broadcast(value);
        tm_coalesced_out[(d - ty) / BLOCK_WIDTH] = value;
    }

    if(tx == 0){
        for(int i = 0; i < D / BLOCK_WIDTH; i++){
            coalesced_out[i * BLOCK_WIDTH + ty] = tm_coalesced_out[i];
        }
    }
    
    __syncthreads();

    // write out
    if(ty == 0){
        for(int d = tx; d < D; d += BLOCK_WIDTH) {
            output[batch * num_blocks_for_head * D + by * D + d] = coalesced_out[d];
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