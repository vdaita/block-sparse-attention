#include <torch/types.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE, int D>
__global__
void forward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    const int* block_indices,
    const int Nq,
    const int Nk, 
    const int B,
    const int block_count,
    const float softmax_scale,
    float* output,
    const bool use_causal_mask
) {
    int tx = threadIdx.x; // Specific query being processed
    int blockSize = blockDim.x; // This specifies how big the given block is
    int bx = blockIdx.x; // Specific block being processed
    int by = blockIdx.y; // Specifies the batch/head currently being processed. The input to this kernel should be reshaped such that the first dimension is B * H

    int q_offset = by * Nq * D + bx * blockSize * D + tx * D;

    extern __shared__ float shared_memory[];
    int tile_size = blockSize * D;
    float* shared_q = shared_memory;
    float* shared_k = &shared_memory[tile_size];
    float* shared_v = &shared_memory[2 * tile_size];

    float running_max = -INFINITY;
    float running_sum = 0;

    float* acc[BLOCK_SIZE][D] = {0};
    float* P[BLOCK_SIZE] = {0};

    for(int j = 0; j < block_count; j++){
        for(int x = 0; x < D; x++){ // should try to introduce coalescing into this loop
            shared_k[tx * D + x] = K[by * Nk * D + block_indices[j] * blockSize * D + tx * D + x];
            shared_v[tx * D + x] = V[by * Nk * D + block_indices[j] * blockSize * D + tx * D + x];
        }
        __syncthreads();

        float new_max = running_max;

        for(int i = 0; i < blockSize; i++){
            float dot_product = 0;
            for(int x = 0; x < D; x++){
                dot_product += Q[q_offset + x] * shared_k[i * D + x];
            }
            dot_product *= softmax_scale;
            if(use_causal_mask){
                if(block_indices[j] * blockSize + i > bx * blockSize + tx){
                    dot_product = -INFINITY;
                }
            }
            P[i] = dot_product;
            new_max = fmaxf(new_max, dot_product);
        }

        float alpha = __exp2(new_max - running_max);
        running_sum *= (1 / alpha);
        for(int i = 0; i < blockSize; i++){
            float token_weight = __exp2(P[i] - new_max);
            running_sum += token_weight;
            for(int x = 0; x < D; x++){
                acc[i][x] *= (1 / alpha);
                acc[i][x] += token_weight * shared_v[i * D + x];
            }
        }

        running_max = new_max;
        __syncthreads();
    }

    for(int i = 0; i < blockSize; i++){
        for(int x = 0; x < D; x++){
            output[by * Nq * D + bx * blockSize * D + i * D + x] = acc[i][x] / running_sum;
        }
    }
}

torch::Tensor block_sparse_attention_forward(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor query_blocks,
    int64_t block_size,
    float dropout_p) {
    
    // the first dimensions should be B, H, T, D
    int B = queries.size(0);
    int H = queries.size(1);
    int T = queries.size(2);
    int D = queries.size(3);

    dim3 gridDim(ceil(T * 1.0f / block_size), B * H, 1);
    dim3 blockDim(block_size, 1, 1);

    auto output = torch::zeros_like(queries);

    forward_kernel<block_size, D><<<gridDim, blockDim>>>(
        queries.data_ptr<float>(),
        keys.data_ptr<float>(),
        values.data_ptr<float>(),
        query_blocks.data_ptr<int>(),
        T,
        T,
        B * H,
        query_blocks.size(0),
        1.0 / sqrtf(D),
        output.data_ptr<float>(),
        false
    );

    return output;
}