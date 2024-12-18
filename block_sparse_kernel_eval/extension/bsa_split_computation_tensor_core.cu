// Idea: use tensor cores to split up the computation across different blocks, and then reduce again
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>

#define D 64
constexpr int BLOCK_SIZE = 16;
using namespace nvcuda;

#define NUM_CHUNKS 4

__global__
void forward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    const int* block_indices,
    const int num_blocks_selected,
    float* output,
    float* output_sum, 
    float* output_max,
    const int T
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int b = blockIdx.y;
    int num_batches = blockDim.y;
    
    int chunk_index = blockIdx.z;

    __shared__ half shared_q[BLOCK_SIZE * 16];
    __shared__ half shared_k[BLOCK_SIZE * 16];
    __shared__ half shared_p[BLOCK_SIZE * 16];
    __shared__ float shared_acc[16 * 16];

    float acc[D] = {0};

    float sum = 0;
    float curr_max = -INFINITY;

    int q_idx = (b * T + bx * BLOCK_SIZE + tx) * D;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_v;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_acc;


    __syncthreads();

    for(int i = chunk_index * (num_blocks_selected / NUM_CHUNKS); i < (chunk_index + 1) * (num_blocks_selected / NUM_CHUNKS); i++){
        int block = block_indices[(b * ((T + BLOCK_SIZE - 1) / BLOCK_SIZE) + bx) * num_blocks_selected + i];
        
        int kv_idx = (b * T + block * BLOCK_SIZE) * D;

        if(tx < BLOCK_SIZE){
          for(int j = 0; j < 16; j++){
            shared_acc[tx * 16 + j] = 0;
          }
        }
        wmma::fill_fragment(frag_acc, 0.0f);

        for(int d_start = 0; d_start < D; d_start += 16){
          if(tx < BLOCK_SIZE){
            for(int d_off = 0; d_off < 16; d_off++){
              shared_q[tx * 16 + d_off] = __float2half(Q[q_idx + d_start + d_off]);
              shared_k[tx * 16 + d_off] = __float2half(K[kv_idx + tx * D + d_start + d_off]);
            }
          }
          wmma::load_matrix_sync(frag_a, shared_q, 16);
          wmma::load_matrix_sync(frag_b, shared_k, 16);
          wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
          __syncthreads();
        }
        wmma::store_matrix_sync(shared_acc, frag_acc, 16, wmma::mem_row_major);

        float new_max = curr_max;
        if(tx < BLOCK_SIZE){
          for(int j = 0; j < BLOCK_SIZE; j++){
            new_max = fmaxf(new_max, shared_acc[tx * 16 + j]);
          }
        }

        __syncthreads();

        if(tx < BLOCK_SIZE){
          float difference = expf(curr_max - new_max);
          sum *= difference;
          for(int d = 0; d < D; d++){
            acc[d] *= difference;
          }
        }

        if(tx < BLOCK_SIZE) {
          for(int j = 0; j < BLOCK_SIZE; j++){
            float adj_attn_weight = expf(shared_acc[tx * 16 + j] - new_max);
            shared_p[tx * 16 + j] = __float2half(adj_attn_weight);
            sum += adj_attn_weight;
          }
        }
        
        wmma::load_matrix_sync(frag_a, shared_p, 16);

        __syncthreads();

        for(int d_start = 0; d_start < D; d_start += 16){
          if(tx < BLOCK_SIZE){
            for(int d_off = 0; d_off < 16; d_off++){
              shared_k[tx * 16 + d_off] = __float2half(V[kv_idx + tx * D + d_start + d_off]);
            }
          }

          wmma::load_matrix_sync(frag_v, shared_k, 16);
          wmma::fill_fragment(frag_acc, 0.0f);
          __syncthreads();  
          wmma::mma_sync(frag_acc, frag_a, frag_v, frag_acc);
          wmma::store_matrix_sync(shared_acc, frag_acc, 16, wmma::mem_row_major);
          __syncthreads();
          if(tx < BLOCK_SIZE){
            for(int d_off = 0; d_off < 16; d_off++){
              acc[d_start + d_off] += shared_acc[tx * 16 + d_off];
            }
          }  
        }  

        curr_max = new_max;
        __syncthreads();
    }

    if(tx < BLOCK_SIZE){
        for(int d = 0; d < D; d++){
            output[(chunk_index * num_batches * T + b * T + bx * BLOCK_SIZE + tx) * D + d] = acc[d];
        }
        output_sum[chunk_index * num_batches * T + b * T + bx * BLOCK_SIZE + tx] = sum;
        output_max[chunk_index * num_batches * T + b * T + bx * BLOCK_SIZE + tx] = curr_max;
    }
}

__global__ void reduction_kernel(
    float* chunked_out,
    float* chunked_sum,
    float* chunked_max,
    float* out,
    int B, 
    int T
) {
    int batch = blockIdx.y;
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int x = batch * T + bx * blockDim.x + tx;
    float local_sum[NUM_CHUNKS] = {0};
    float local_max[NUM_CHUNKS] = {-INFINITY};
    float acc[D] = {0};

    float global_sum = 0;
    float global_max = -INFINITY;

    for(int i = 0; i < NUM_CHUNKS; i++){
        local_sum[i] = chunked_sum[i * B * T + x]; // <- this is in terms of the local max
        local_max[i] = chunked_max[i * B * T + x];
        global_max = fmaxf(global_max, local_max[i]);
    }

    for(int i = 0; i < NUM_CHUNKS; i++){
      float exp_diff = expf(local_max[i] - global_max);
      global_sum += local_sum[i] * exp_diff;  
      for(int d = 0; d < D; d++){
          acc[d] += exp_diff * chunked_out[i * B * T * D + x * D + d]; // sacle donw the values correspondifnyl 
      }
    }

    for(int i = 0; i < D; i++){
        out[x * D + i] = acc[i] / global_sum;
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

    dim3 forwardGridDim((T + BLOCK_SIZE - 1) / BLOCK_SIZE, B, NUM_CHUNKS);
    dim3 forwardBlockDim(32); // probably would be a good idea to make the block sizes a bit larger, and splitting computation being performed in terms of dimension size

    auto output = torch::zeros_like(queries);

    float* Q = queries.data_ptr<float>();
    float* K = keys.data_ptr<float>();
    float* V = values.data_ptr<float>();
    int* QB_ptr = query_blocks.data_ptr<int>();
    float* O = output.data_ptr<float>();

    auto chunked_output = torch::zeros({NUM_CHUNKS, B, T, D}, queries.options()).contiguous();
    float* chunked_output_ptr = chunked_output.data_ptr<float>();
    
    auto chunked_max = torch::zeros({NUM_CHUNKS, B, T}, queries.options()).contiguous();
    float* chunked_max_ptr = chunked_max.data_ptr<float>();

    auto chunked_sum = torch::zeros({NUM_CHUNKS, B, T}, queries.options()).contiguous();
    float* chunked_sum_ptr = chunked_sum.data_ptr<float>();

    forward_kernel<<<forwardGridDim, forwardBlockDim>>>(Q, K, V, QB_ptr, num_blocks_selected, chunked_output_ptr, chunked_sum_ptr, chunked_max_ptr, T);

    dim3 reductionGridDim((T + BLOCK_SIZE - 1) / BLOCK_SIZE, B);
    dim3 reductionBlockDim(BLOCK_SIZE);

    reduction_kernel<<<reductionGridDim, reductionBlockDim>>>(chunked_output_ptr, chunked_sum_ptr, chunked_max_ptr, O, B, T);
    return output;
}