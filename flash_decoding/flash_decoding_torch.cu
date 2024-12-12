#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdio.h>
#include "flash_decoding.cuh"

torch::Tensor forward(
    torch::Tensor query,
    torch::Tensor keys,
    torch::Tensor values
){
    int B = query.size(0);
    int T = keys.size(1);

    int num_blocks_per_head = min((T + BLOCK_TOKENS - 1) / BLOCK_TOKENS, MAX_NUM_BLOCKS);
    dim3 gridDim(1, num_blocks_per_head, B);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_TOKENS, 1);

    float* query_ptr = query.data_ptr<float>();
    float* keys_ptr = keys.data_ptr<float>();
    float* values_ptr = values.data_ptr<float>();

    auto output_temp = torch::zeros({B, num_blocks_per_head, D}, torch::device(torch::kCUDA));
    auto output_sum_temp = torch::zeros({B, D}, torch::device(torch::kCUDA));
    auto output_max_temp = torch::zeros({B, D}, torch::device(torch::kCUDA));

    float* output_temp_ptr = output_temp.data_ptr<float>();
    float* output_sum_ptr = output_sum_temp.data_ptr<float>();
    float* output_max_ptr = output_max_temp.data_ptr<float>(); 

    auto output = torch::zeros({B, D}, torch::device(torch::kCUDA));
    float* output_ptr = output.data_ptr<float>();

    shared_split_k_kernel<<<gridDim, blockDim>>>(query_ptr, 
        keys_ptr, 
        values_ptr, 
        output_temp_ptr,
        output_sum_ptr,
        output_max_ptr,
        B,
        T
    );

    dim3 gridDimReduction(1, 1, B);
    dim3 blockDimReduction(D, num_blocks_per_head, 1);
    reduction_kernel<<<gridDimReduction, blockDimReduction>>>(
        output_temp_ptr,
        output_sum_ptr,
        output_max_ptr,
        output_ptr,
        num_blocks_per_head
    );
    
    return output;
}