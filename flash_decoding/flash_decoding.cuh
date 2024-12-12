#include <cuda_runtime.h>
// #include <torch/extension.h>
#include <stdio.h>

#define D 64
#define BLOCK_WIDTH 32 // perform some type of calculation
#define BLOCK_TOKENS 32
#define MAX_NUM_BLOCKS 2

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
    float* output_sum,
    float* output_max,
    const int B,
    const int T
) {
    int tx = threadIdx.x; // represents the hidden dim dimension
    int ty = threadIdx.y; // represents the key index dimension
    int by = blockIdx.y; // represents the key index dimension, but which current block
    int num_blocks_per_head = gridDim.y;

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
    float acc = 0.0f;

    for(int i = start_token_kv; i < T; i += BLOCK_TOKENS * num_blocks_per_head){ // this means that T must be padded to the nearest multiple of 32
        // load the right token for this dimension
        // printf("%d %d %d processing idx: %d\n", by, tx, ty, i);
        acc = 0.0f;
        for(int d = tx; d < D; d += BLOCK_WIDTH){
            // printf("%d %d %d processing dimension: %d\n", by, tx, ty, d);
            acc += shared_q[d] * K[batch * T * D + i * D + d]; // d is related to tx, so memory accesses should be coalesced
        }
        __syncwarp();
        acc = warp_add_and_broadcast(acc);
        // printf("%d %d %d acc: %f\n", by, tx, ty, acc);

        float prev_max_qk = max_qk;
        max_qk = fmaxf(acc, max_qk);
        float alpha = expf(prev_max_qk - max_qk);
        float normalized_weight = expf(acc - max_qk);
        sum_qk = sum_qk * alpha + normalized_weight;

        // now that the accumulator has the weight for the entire thing
        for(int di = 0; di < D / BLOCK_WIDTH; di++){
            int d = tx + di * BLOCK_WIDTH;
            values[di] = values[di] * alpha + normalized_weight * V[batch * T * D + i * D + d];
            // printf("%d %d %d adding value to dim: %d %f with alpha %f and nw %f\n", by, tx, ty, d, values[di], alpha, normalized_weight);
        }
    }
    for(int i = 0; i < D / BLOCK_WIDTH; i++){
        shared_out[ty][tx + i * BLOCK_WIDTH] = values[i];
    }

    // printf("%d %d %d max: %f\n", by, tx, ty, max_qk);
    // printf("%d %d %d sum: %f\n", by, tx, ty, sum_qk);

    if(tx == 0){
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
    __syncwarp();

    float alpha = expf(local_max_qk - max_qk); // expf curr_max - new_max
    sum_qk *= alpha;
    __syncwarp();

    sum_qk = warp_add_and_broadcast(sum_qk); // now, each warp has the same sum from adding up all of the sums adjusted by alpha

    // printf("Global values: tx %d ty %d local_max %f global_max %f alpha %f sum %f\n", tx, ty, local_max_qk, max_qk, alpha, sum_qk);

    float tm_coalesced_out[D / BLOCK_WIDTH];

    // each warp handles 32 separate tokens for a given dimension
    // meaning each thread must handle the 4 dimensions
    for(int d = ty; d < D; d += BLOCK_WIDTH){
        float value = shared_out[tx][d] * alpha;
        // add the values together
        value = warp_add_and_broadcast(value);
        __syncwarp();
        tm_coalesced_out[d / BLOCK_WIDTH] = value;
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
            printf("batch=%d, nbph=%d, by=%d, d=%d, val=%f\n", batch, num_blocks_per_head, by, d, coalesced_out[d]);
            output[batch * num_blocks_per_head * D + by * D + d] = coalesced_out[d];
        }
        if(tx == 0){
            printf("batch=%d, by=%d, sum_qk=%f, max_qk=%f\n", batch, by, sum_qk, max_qk);
            output_sum[batch * num_blocks_per_head + by] = sum_qk;
            output_max[batch * num_blocks_per_head + by] = max_qk;
        }
    }
}

__global__
void reduction_kernel (
    __const__ float* output,
    __const__ float* output_sum,
    __const__ float* output_max,
    float* result,
    int num_blocks_per_head
) {
    int tx = threadIdx.x; // D
    int ty = threadIdx.y; // at most 8
    int batch = blockIdx.z;

    __shared__ float shared_acc[MAX_NUM_BLOCKS][D];
    __shared__ float shared_sum[MAX_NUM_BLOCKS];
    __shared__ float shared_max[MAX_NUM_BLOCKS];

    if(ty < num_blocks_per_head){
        printf("Loading value into shared mem: ty=%d tx=%d %f\n", ty, tx, output[batch * num_blocks_per_head * D + ty * D + tx]);
        shared_acc[ty][tx] = output[batch * num_blocks_per_head * D + ty * D + tx];
        if(tx == 0){
            shared_sum[ty] = output_sum[batch * num_blocks_per_head + ty];
            shared_max[ty] = output_max[batch * num_blocks_per_head + ty];
        }
    } else {
        shared_acc[ty][tx] = 0;
        if(tx == 0){
            shared_sum[ty] = 0;
            shared_max[ty] = -INFINITY;
        }
    }

    for(int stride = (num_blocks_per_head + 1) / 2; stride > 0; stride /= 2){
        printf("Stride: %d\n", stride);
        __syncthreads();
        float new_max = -INFINITY;
        float alpha = 0;
        float beta = 0;
        float right_acc = 0;
        float right_sum = 0;

        // if(ty + stride < MAX_NUM_BLOCKS){
        if(ty < stride){
            // look at ty, ty + stride
            printf("Adding %d and %d together at x dimension value %d\n", ty, ty + stride, tx);
            new_max = fmaxf(shared_max[ty], shared_max[ty + stride]);
            alpha = expf(shared_max[ty] - new_max);
            beta = expf(shared_max[ty + stride] - new_max);
            right_acc = shared_acc[ty + stride][tx];
            printf("Shared mem value of %d+%d=%d at dim %d, value %f\n", ty, stride, ty + stride, tx, shared_acc[ty + stride][tx]);
            right_sum = shared_sum[ty + stride];
        }
        __syncthreads();
        // if(ty + stride < MAX_NUM_BLOCKS){
        if(ty < stride){
            printf("Reducing with shared max %f, shift shared max %f, new max %f, alpha %f, beta %f, right_acc %f, right_sum %f, curr_acc %f, curr_sum %f\n", shared_max[ty], shared_max[ty + stride], new_max, alpha, beta, right_acc, right_sum, shared_acc[ty][tx], shared_sum[ty]);
            shared_sum[ty] = alpha * shared_sum[ty] + beta * right_sum;
            shared_acc[ty][tx] = alpha * shared_acc[ty][tx] + beta * right_acc;
            shared_max[ty] = new_max;
        }
    }

    __syncthreads(); //MAYBE: threads do not need to be synced up because everything we care about is with ty = 0, tx = tx which we just set
    if(ty == 0){
        result[batch * D + tx] = shared_acc[ty][tx] / shared_sum[ty];
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