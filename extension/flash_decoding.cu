#include <cuda_runtime.h>
#include <torch/extension.h>

#define D 128
#define BLOCK_WIDTH 32 // perform some type of calculation
#define BLOCK_TOKENS 32

__device__ float get_warp_max(float val) {
    // Full mask for all 32 threads in a warp
    unsigned int mask = 0xffffffff;

    // Iteratively reduce using XOR to exchange data
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_xor_sync(mask, val, offset));
    }
    return val;  // Maximum value within the warp
}

__device__ warp_add_and_broadcast() {

}

__global__
void shared_split_k_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    const int B,
    const int T,
    const int num_to_process
) {
    int tx = threadIdx.x; // represents the hidden dim dimension
    int ty = threadIdx.y; // represents the key index dimension
    int by = blockIdx.y; // represents the key index dimension, but which current block
    int num_blocks_for_head = gridDim.y;

    int batch = blockIdx.z;

    int start_token_kv = by * BLOCK_TOKENS + ty;

    // load data into shared memory
    __shared__ float shared_q[D];
    for(int i = ty * BLOCK_WIDTH + tx; i < D; i += BLOCK_WIDTH * BLOCK_HEIGHT){
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
        acc = warp_add_and_broadcast(add); // TODO: implement

        // TODO: make sure you scale the number down
        float alpha = expf(max_qk - acc);
        max_qk = fmaxf(acc, max_qk);

        // now that the accumulator has the weight for the entire thing
        for(int di = 0; di < 4; di++){
            int d = tx + di * BLOCK_WIDTH;
            values[di] = acc * V[batch * T * D + i * D + d];
        }
    }
    for(int i = 0; i < D / BLOCK_WIDTH; i++){
        shared_out[ty][tx + i * BLOCK_WIDTH] = values[i];
    }

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
    max_qk = get_warp_max(max_qk);
    float alpha = expf(local_max_qk - max_qk);
    sum_qk *= alpha;
    sum_qk = warp_add_and_broadcast(value); // now, each warp has the same sum from adding up all of the sums adjusted by alpha

    // each warp handles 32 separate tokens for a given dimension
    // meaning each thread must handle the 4 dimensions
    for(int d = ty; d < D; d += BLOCK_WIDTH){
        float value = shared_out[tx][y] * alpha;
        // add the values together
        value = warp_add_and_broadcast(value);
        if(tx == 0){
            coalesced_out[d] = value; // save!
        }
    }
    __syncthreads();

    // write out
    if(ty == 0){
        for(int d = tx; d < D; d += BLOCK_WIDTH) {
            output[batch * num_blocks_for_head * D + by * D + d] = shared_out[0][d];
        }
    }
}

torch::Tensor forward(){

}