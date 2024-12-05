#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>

#define D 64
constexpr int BLOCK_SIZE = 16;
const int block_batch_size = 4;

using namespace nvcuda;

// Unfinished, not going to work

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
    int ty = threadIdx.y;
    
    int block_batch_offset = ty * block_batch_size;

    int bx = blockIdx.x;
    int b = blockIdx.y;

    __shared__ half shared_q[num_batched_blocks * BLOCK_SIZE * 16];
    __shared__ half shared_k[num_batched_blocks * BLOCK_SIZE * 16];
    __shared__ float shared_p[num_batched_blocks * 16 * 16];
    __shared__ float shared_acc[num_batched_blocks * 16 * 16];

    float acc[D] = {0};

    float sum = 0;
    float curr_max = -INFINITY;

    int q_idx = (b * T + bx * BLOCK_SIZE + tx) * D;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_acc;

    __syncthreads();

    for(int i = 0; i < num_blocks_selected; i++){
        int block = block_indices[(b * ((T + BLOCK_SIZE - 1) / BLOCK_SIZE) + bx) * num_blocks_selected + i];
        
        int kv_idx = (b * T + block * BLOCK_SIZE) * D;

        if(tx < BLOCK_SIZE){
          for(int j = 0; j < 16; j++){
            shared_p[block_batch_offset + tx * 16 + j] = 0;
          }
        }
        wmma::fill_fragment(frag_acc, 0.0f);

        for(int d_start = 0; d_start < D; d_start += 16){
          if(tx < BLOCK_SIZE){
            for(int d_off = 0; d_off < 16; d_off++){
              shared_q[block_batch_offset + tx * 16 + d_off] = __float2half(Q[q_idx + d_start + d_off]);
              // shared_k[d_off * 16 + tx] = __float2half(K[kv_idx + tx * D + d_start + d_off]);
              shared_k[block_batch_offset + tx * 16 + d_off] = __float2half(K[kv_idx + tx * D + d_start + d_off]);
            }
          }

          // __syncthreads();
          wmma::load_matrix_sync(frag_a, shared_q + block_batch_offset, 16);
          // __syncthreads();
          wmma::load_matrix_sync(frag_b, shared_k, 16);
          // __syncthreads();
          wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
          // if(tx < BLOCK_SIZE){
          //   for(int j = 0; j < 16; j++){
          //     float weight = 0;
          //     for(int d_off = 0; d_off < 16; d_off++){
          //       weight += __half2float(__hmul(shared_q[tx * 16 + d_off], shared_k[d_off * 16 + j]));
          //     }
          //     [tx * 16 + j] += weight;
          //   }
          // }

          __syncthreads();
        }

        wmma::store_matrix_sync(shared_p, frag_acc, 16, wmma::mem_row_major);
        float new_max = curr_max;
        if(tx < BLOCK_SIZE){
          for(int j = 0; j < BLOCK_SIZE; j++){
            new_max = fmaxf(new_max, shared_p[tx * 16 + j]);
          }
        }

        __syncthreads();

        if(tx < BLOCK_SIZE){
          float difference = expf(curr_max - new_max);
          sum *= difference;
          for(int d = 0; d < D; d++){
              acc[d] *= difference;
          }

          for(int j = 0; j < BLOCK_SIZE; j++){
              float norm_weight = expf(shared_p[tx * 16 + j] - new_max);
              for(int d = 0; d < D; d++){
                  acc[d] += norm_weight * V[kv_idx + j * D + d];
              }
              sum += norm_weight;
          }

          curr_max = new_max;
        }
        __syncthreads();
    }

    if(ty == 0){ // assuming that the biggest bottleneck is the matrix multiplication and not the reduction afterwards
        // current max is set to the current maximum for this thread
        for(int batch = 1; batch < (num_blocks_selected / block_batch_size); batch++){
            
        }
    }

    if(tx < BLOCK_SIZE){
      int out_idx = (b * T + bx * BLOCK_SIZE + tx) * D;
      for(int d = 0; d < D; d++){
          output[out_idx + d] = acc[d] / sum;
      }
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
    dim3 blockDim(32, num_blocks_selected);

    auto output = torch::zeros_like(queries);

    float* Q = queries.data_ptr<float>();
    float* K = keys.data_ptr<float>();
    float* V = values.data_ptr<float>();
    int* QB_ptr = query_blocks.data_ptr<int>();
    float* O = output.data_ptr<float>();

    forward_kernel<<<gridDim, blockDim>>>(Q, K, V, QB_ptr, num_blocks_selected, O, T);

    return output;
}