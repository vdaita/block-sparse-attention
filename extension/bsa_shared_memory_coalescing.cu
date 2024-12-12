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
    float* output,
    const int T
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int b = blockIdx.y;

    __shared__ float shared_q[BLOCK_SIZE][D];
    __shared__ float shared_k[BLOCK_SIZE][D];
    __shared__ float shared_v[BLOCK_SIZE][D];

    float P[BLOCK_SIZE];
    float acc[D] = {0};

    float sum = 0;
    float curr_max = -INFINITY;

    int q_idx = (b * T + bx * BLOCK_SIZE + tx) * D;
    // for(int d = 0; d < D; d++){
    //     shared_q[tx][d] = Q[q_idx + d];
    // }
    // each thread should indicate a dimension
    for(int i = 0; i < BLOCK_SIZE; i++){
        shared_q[i][tx] = (b * T + bx * BLOCK_SIZE + i) * D + tx;
    }

    __syncthreads();

    for(int i = 0; i < num_blocks_selected; i++){
        int block = block_indices[(b * ((T + BLOCK_SIZE - 1) / BLOCK_SIZE) + bx) * num_blocks_selected + i];
        int kv_idx = (b * T + block * BLOCK_SIZE + tx) * D;
        // for(int d = 0; d < D; d++){
        //     shared_k[tx][d] = K[kv_idx + d];
        //     shared_v[tx][d] = V[kv_idx + d];
        // }
        for(int j = 0; j < BLOCK_SIZE; j++){
            shared_k[j][tx] = K[(b * T + block * BLOCK_SIZE + j) * D + tx];
            shared_v[j][tx] = V[(b * T + block * BLOCK_SIZE + j) * D + tx];
        }
        __syncthreads();

        float new_max = curr_max;
        for(int j = 0; j < BLOCK_SIZE; j++){
            float weight = 0;
            for(int d = 0; d < D; d++){
                weight += shared_q[tx][d] * shared_k[j][d];
            }
            new_max = fmaxf(new_max, weight);
            P[j] = weight;
        }

        __syncthreads();

        float difference = expf(curr_max - new_max);
        sum *= difference;
        for(int d = 0; d < D; d++){
            acc[d] *= difference;
        }

        for(int j = 0; j < BLOCK_SIZE; j++){
            float norm_weight = expf(P[j] - new_max);
            for(int d = 0; d < D; d++){
                acc[d] += norm_weight * shared_v[j][d];
            }
            sum += norm_weight;
        }

        curr_max = new_max;

        __syncthreads();
    }

    int out_idx = (b * T + bx * BLOCK_SIZE + tx) * D;
    for(int d = 0; d < D; d++){
        output[out_idx + d] = acc[d] / sum;
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

    dim3 gridDim((T + BLOCK_SIZE - 1) / BLOCK_SIZE, B);
    dim3 blockDim(BLOCK_SIZE);

    auto output = torch::zeros_like(queries);

    float* Q = queries.data_ptr<float>();
    float* K = keys.data_ptr<float>();
    float* V = values.data_ptr<float>();
    int* QB_ptr = query_blocks.data_ptr<int>();
    float* O = output.data_ptr<float>();

    forward_kernel<<<gridDim, blockDim>>>(Q, K, V, QB_ptr, num_blocks_selected, O, T);

    return output;
}