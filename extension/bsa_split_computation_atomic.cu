#include <cuda_runtime.h>
#include <torch/extension.h>

#define D 64
#define NUM_CHUNKS 4
constexpr int BLOCK_SIZE = 16;

__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

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
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int b = blockIdx.y;

    float P[BLOCK_SIZE];
    float acc[D] = {0};
    __shared__ float shared_acc[BLOCK_SIZE][D];
    __shared__ float shared_max[BLOCK_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE];

    // if(ty == 0){
    for(int d = (D / NUM_CHUNKS) * ty; d < (D / NUM_CHUNKS) * (ty + 1); d++){
        shared_acc[tx][d] = 0;
    }

    if(ty == 0){
        shared_max[tx] = -INFINITY;
        shared_sum[tx] = 0;
    }

    float sum = 0;
    float curr_max = -INFINITY;

    int q_idx = (b * T + bx * BLOCK_SIZE + tx) * D;

    for(int i = ty * (num_blocks_selected / NUM_CHUNKS); i < (ty + 1) * (num_blocks_selected / NUM_CHUNKS); i++){
        int block = block_indices[(b * num_blocks + bx) * num_blocks_selected + i];
        float new_max = curr_max;
        for(int j = 0; j < BLOCK_SIZE; j++){
            float weight = 0;
            for(int d = 0; d < D; d++){
                weight += Q[q_idx + d] * K[(b * T * D) + (block * BLOCK_SIZE * D) + (j * D) + d];
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
                acc[d] += norm_weight * V[(b * T * D) + (block * BLOCK_SIZE * D) + (j * D) + d];
            }
            sum += norm_weight;
        }

        curr_max = new_max;
    }

    atomicMaxFloat(&shared_max[tx], curr_max);
    __syncthreads();


    for(int d = 0; d < D; d++){
        int sd = (d + ty * (D / NUM_CHUNKS)) % D;
        atomicAdd(&shared_acc[tx][sd], acc[sd] * expf(curr_max - shared_max[tx]));
    }
    atomicAdd(&shared_sum[tx], sum * expf(curr_max - shared_max[tx])); // how do I monkey patch everything?

    __syncthreads();

    // if(ty == 0){
    int out_idx = (b * T + bx * BLOCK_SIZE + tx) * D;
    for(int d = (ty) * (D / NUM_CHUNKS); d < D; d++){
        output[out_idx + d] = shared_acc[tx][d] / shared_sum[tx];
    }
    // }
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
    dim3 blockDim(BLOCK_SIZE, NUM_CHUNKS);

    auto output = torch::zeros_like(queries);

    float* Q = queries.data_ptr<float>();
    float* K = keys.data_ptr<float>();
    float* V = values.data_ptr<float>();
    int* QB_ptr = query_blocks.data_ptr<int>();
    float* O = output.data_ptr<float>();

    forward_kernel<<<gridDim, blockDim>>>(Q, K, V, QB_ptr, num_blocks_selected, num_blocks, O, T);

    return output;
}