#include <cuda_runtime.h>
#include <torch/extension.h>

#define D 64
constexpr int BLOCK_SIZE = 32;

__global__
void forward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    const int* block_indices,
    const int num_blocks_selected,
    const int num_blocks,
    float* output,
    const int T
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int b = blockIdx.y;

    float P[BLOCK_SIZE];
    float acc[D] = {0};

    float sum = 0;
    float curr_max = -INFINITY;

    // int q_idx = (b * T + bx * BLOCK_SIZE + tx) * D;
    
    __shared__ float shared_q[D][BLOCK_SIZE];
    for(int d = 0; d < D; d++){
        shared_q[d][tx] = Q[(b * T * D) + (d * T) + (bx * BLOCK_SIZE + tx)];
    }

    __syncthreads();

    for(int i = 0; i < num_blocks_selected; i++){
        int block = block_indices[(b * num_blocks + bx) * num_blocks_selected + i];

        float new_max = curr_max;
        for(int j = 0; j < BLOCK_SIZE; j++){
            float weight = 0;
            for(int d = 0; d < D; d++){
                weight += shared_q[d][tx] * K[(b * T * D) + (d * T) + (block * BLOCK_SIZE + j)];
            }
            new_max = fmaxf(new_max, weight);
            P[j] = weight;
        }

        float difference = expf(curr_max - new_max);
        sum *= difference;
        for(int d = 0; d < D; d++){
            acc[d] *= difference;
        }

        for(int j = 0; j < BLOCK_SIZE; j++){
            float norm_weight = expf(P[j] - new_max);
            for(int d = 0; d < D; d++){
                acc[d] += norm_weight * V[(b * T * D) + (d * T) + (block * BLOCK_SIZE + j)];
            }
            sum += norm_weight;
        }

        curr_max = new_max;
    }

    for(int d = 0; d < D; d++){
        output[b * T * D + d * T + (bx * BLOCK_SIZE + tx)] = acc[d] / sum;
    }
}

torch::Tensor forward(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor query_blocks
) {
    int B = queries.size(0);
    int T = queries.size(1);
    // D should match the macro D
    int num_blocks_selected = query_blocks.size(2);
    int num_blocks = query_blocks.size(1);

    dim3 gridDim((T + BLOCK_SIZE - 1) / BLOCK_SIZE, B);
    dim3 blockDim(BLOCK_SIZE);
    torch::Tensor queries_transposed = queries.transpose(1, 2).contiguous();
    torch::Tensor keys_transposed = keys.transpose(1, 2).contiguous();
    torch::Tensor values_transposed = values.transpose(1, 2).contiguous();


    auto output_transposed = torch::zeros_like(queries_transposed);

    float* Q = queries_transposed.data_ptr<float>();
    float* K = keys_transposed.data_ptr<float>();
    float* V = values_transposed.data_ptr<float>();
    int* QB_ptr = query_blocks.data_ptr<int>();
    float* O_T = output_transposed.data_ptr<float>();

    forward_kernel<<<gridDim, blockDim>>>(Q, K, V, QB_ptr, num_blocks_selected, num_blocks, O_T, T);

    cudaDeviceSynchronize();
    auto output = output_transposed.transpose(1, 2);

    return output;
}