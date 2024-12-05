#include <torch/extension.h>
#include <torch/script.h>
#include <mma.h>
#include <cuda_fp16.h>

#define D 128
constexpr int BLOCK_SIZE = 16;
using namespace nvcuda;

__global__
void forward_kernel(
    const half* Q,
    const half* K,
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

    float acc[D] = {0};

    float sum = 0;
    float curr_max = -INFINITY;

    int q_idx = (b * T + bx * BLOCK_SIZE + tx) * D;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    __shared__ half sharedA[BLOCK_SIZE * 16];
    __shared__ half sharedB[BLOCK_SIZE * 16];

    for(int i = 0; i < num_blocks_selected; i++){
        int block = block_indices[(b * num_blocks + bx) * num_blocks_selected + i];
        int kv_idx = (b * T + block * BLOCK_SIZE + tx) * D;

        float new_max = curr_max;
        wmma::fill_fragment(acc_frag, 0.0f);

        for(int q_chunk_start = 0; q_chunk_start < D; q_chunk_start += 16){
          // load the query chunk
          for(int dq = q_chunk_start; dq < q_chunk_start + 16; dq++){
            sharedA[tx * 16 + (dq - q_chunk_start)] = Q[q_idx + dq];
          }
          __syncthreads();
          wmma::load_matrix_sync(a_frag, sharedA, 16);
          for(int k_chunk_start = 0; k_chunk_start < D; k_chunk_start += 16){
            // load the kv chunk
            for(int dk = k_chunk_start; dk < k_chunk_start + 16; dk++){
              // sharedB[tx * 16 + (dk - k_chunk_start)] = K[kv_idx + dk];
              // transpose as you load into sharedB
              sharedB[(dk - k_chunk_start) * 16 + tx] = K[kv_idx + dk];
            }
            __syncthreads();
            wmma::load_matrix_sync(b_frag, sharedB, 16);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
          }
        }

        for(int j = 0; j < BLOCK_SIZE; j++){
          new_max = fmaxf(new_max, acc_frag.x[tx * 16 + j]);
        }

        float difference = expf(curr_max - new_max);
        sum *= difference;
        for(int d = 0; d < D; d++){
            acc[d] *= difference;
        }

        for(int j = 0; j < BLOCK_SIZE; j++){
            float norm_weight = expf(acc_frag.x[tx * 16 + j] - new_max);
            for(int d = 0; d < D; d++){
                acc[d] += norm_weight * V[kv_idx + d];
            }
            sum += norm_weight;
        }

        curr_max = new_max;
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
    int num_blocks = query_blocks.size(1);

    dim3 gridDim((T + BLOCK_SIZE - 1) / BLOCK_SIZE, B);
    dim3 blockDim(BLOCK_SIZE);

    auto output = torch::zeros_like(queries);

    torch::Tensor queries_half = queries.to(torch::kHalf);
    torch::Tensor keys_half = keys.to(torch::kHalf);
    torch::Tensor values_half = values.to(torch::kHalf);

    half* Q = (half*) queries_half.data_ptr<at::Half>();
    half* K = (half*) keys_half.data_ptr<at::Half>();
    float* V = values.data_ptr<float>();
    int* QB_ptr = query_blocks.data_ptr<int>();
    float* O = output.data_ptr<float>();

    forward_kernel<<<gridDim, blockDim>>>(Q, K, V, QB_ptr, num_blocks_selected, num_blocks, O, T);

    return output;
}