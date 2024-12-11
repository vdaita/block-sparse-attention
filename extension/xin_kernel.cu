#ifndef __FLASHATTENTION2_CUH__
#define __FLASHATTENTION2_CUH__

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

/* Adopted from https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu */

/*
    Constraints for block:
    - Br = Bc <= WARP_SIZE
    - Br = Bc \in {16, 32}
    - Dim % 16 = 0
*/

const int Br = 16;
const int Bc = 16;
const int Dim = 64;  // GPT-2

/*
    Constraints for WMMA:
    - tf32,tf32->float: 16x16x8
    - half,half->float: 16x16x16
*/

const int WARP_SIZE = 32;

/* FlashAttention2 kernel for TF32 inputs with causal inference */
__global__ void flashattn2_tf32_kernel(float *O, const float* Q, const float* K, const float* V,
                                       const int N, const int Tr, const int Tc,
                                       const float softmax_scale) {
    // by: batch index, bz: head index
    int by = blockIdx.y, bz = blockIdx.z;
    int qkv_offset = by * gridDim.z * N * Dim + bz * N * Dim;

    // bx * Br + tx: row index
    int bx = blockIdx.x, tx = threadIdx.x;

    // shared memory for Q,K,V,S
    __shared__ float sram[4 * Br * Dim + Br * Bc];

    const int tile_size = Br * Dim;
    float* Qi = sram;
    float* Kj = sram + tile_size;
    float* Vj = sram + tile_size * 2;
    float* Oi = sram + tile_size * 3;
    float* Sij = sram + tile_size * 4;

    int cur_Br = min(Br, N - bx * Br);

    // initialize l and m
    float row_l = 0, row_m = -INFINITY;

    // load Qi to SRAM
    for (int x = 0; x < cur_Br * Dim; x += WARP_SIZE) {
        if (x + tx < cur_Br * Dim) {
            Qi[x + tx] = Q[qkv_offset + bx * tile_size + x + tx];
            Oi[x + tx] = 0;
        }
    }

    for (int j = 0; j < Tc; ++j) {

        if (j > bx) continue;

        // causal mask
        int cur_Bc = min(Bc, N - j * Bc);
        int row_idx = bx * Br + tx;
        int col_idx = j * Bc;
        int max_c = min(cur_Bc, row_idx - col_idx + 1);

        // load Kj,Vj to SRAM: coalesced access
        for (int x = 0; x < cur_Bc * Dim; x += WARP_SIZE) {
            if (x + tx < cur_Bc * Dim) {
                Kj[x + tx] = K[qkv_offset + j * tile_size + x + tx];
                Vj[x + tx] = V[qkv_offset + j * tile_size + x + tx];
            }
        }
        __syncthreads();

        // record l,m
        float prev_row_m = row_m;
        float prev_row_l = row_l;

        // compute Sij = Qi * Kj^T
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> mat_q;
        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> mat_k;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> mat_s;

        for (int x = 0; x < Br; x += 16) {
            for (int y = 0; y < Bc; y += 16) {
                wmma::fill_fragment(mat_s, 0.0f);
                for (int k = 0; k < Dim; k += 8) {
                    wmma::load_matrix_sync(mat_q, Qi + x * Dim + k, Dim);
                    wmma::load_matrix_sync(mat_k, Kj + y * Dim + k, Dim);
                    wmma::mma_sync(mat_s, mat_q, mat_k, mat_s);
                }

                // store to Sij
                wmma::store_matrix_sync(Sij + x * Bc + y, mat_s, Bc, wmma::mem_row_major);
            }
        }

        // row_m = max(S), row_l = rowsum(exp(S - row_m))
        float new_row_m = -INFINITY;
        float new_row_l = 0;
        if (tx < cur_Br) {
            for (int c = 0; c < max_c; ++c) {
                Sij[tx * Bc + c] = Sij[tx * Bc + c] * softmax_scale;
                new_row_m = max(new_row_m, Sij[tx * Bc + c]);
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            for (int c = 0; c < max_c; ++c) {
                Sij[tx * Bc + c] = __expf(Sij[tx * Bc + c] - new_row_m);
                new_row_l += Sij[tx * Bc + c];
            }

            // prepare input
            for (int c = 0; c < Bc; ++c) {
                if (c < max_c) {
                    Sij[tx * Bc + c] = wmma::__float_to_tf32(Sij[tx * Bc + c]);
                } else {  // zero out
                    Sij[tx * Bc + c] = 0;
                }
            }
        }

        // compute Sij * Vj and store to Kj
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> mat_s2;
        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> mat_v;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> mat_o;

        for (int x = 0; x < Br; x += 16) {
            for (int y = 0; y < Dim; y += 16) {
                wmma::fill_fragment(mat_o, 0.0f);
                for (int k = 0; k < Bc; k += 8) {
                    wmma::load_matrix_sync(mat_s2, Sij + x * Bc + k, Bc);
                    wmma::load_matrix_sync(mat_v, Vj + k * Dim + y, Dim);
                    wmma::mma_sync(mat_o, mat_s2, mat_v, mat_o);
                }
                wmma::store_matrix_sync(Kj + x * Dim + y, mat_o, Dim, wmma::mem_row_major);
            }
        }

        // update Oi
        if (tx < cur_Br) {
            row_m = max(prev_row_m, new_row_m);

            float var1 = __expf(prev_row_m - row_m) * prev_row_l;
            float var2 = __expf(new_row_m - row_m);
            row_l = var1 + var2 * new_row_l;

            float row_l_inv = 1 / row_l;
            for (int x = 0; x < Dim; ++x) {
                Oi[tx * Dim + x] = row_l_inv * (var1 * Oi[tx * Dim + x] + var2 * Kj[tx * Dim + x]);
            }
        }
        __syncthreads();
    }

    // write Oi to global memory
    for (int x = 0; x < cur_Br * Dim; x += WARP_SIZE) {
        if (x + tx < cur_Br * Dim) {
            O[qkv_offset + bx * tile_size + x + tx] = Oi[x + tx];
        }
    }
}

/* FlashAttention kernel for HALF inputs with causal inference */
__global__ void flashattn2_half_kernel(float *O, const half* Q, const half* K, const half* V,
                                       const int N, const int Tr, const int Tc,
                                       const float softmax_scale) {
    // by: batch index, bz: head index
    int by = blockIdx.y, bz = blockIdx.z;
    int qkv_offset = by * gridDim.z * N * Dim + bz * N * Dim;

    // bx * Br + tx: row index
    int bx = blockIdx.x, tx = threadIdx.x;

    // shared memory for Q,K,V,S
    __shared__ half sram[7 * Br * Dim + 3 * Br * Bc];

    const int tile_size = Br * Dim;
    half* Qi_half = sram;
    half* Kj_half = sram + tile_size;
    half* Vj_half = sram + tile_size * 2;
    half* Sij_half = sram + tile_size * 3;

    float* Sij = (float *)(sram + tile_size * 3 + Br * Bc);
    float* _Oi = (float *)(sram + tile_size * 3 + Br * Bc * 3);
    float* Oi = (float *)(sram + tile_size * 5 + Br * Bc * 3);

    int cur_Br = min(Br, N - bx * Br);

    // initialize l and m
    float row_l = 0, row_m = -INFINITY;

    // load Qi to SRAM
    for (int x = 0; x < cur_Br * Dim; x += WARP_SIZE) {
        if (x + tx < cur_Br * Dim) {
            Qi_half[x + tx] = Q[qkv_offset + bx * tile_size + x + tx];
            Oi[x + tx] = 0;
        }
    }

    for (int j = 0; j < Tc; ++j) {

        if (j > bx) continue;

        // causal mask
        int cur_Bc = min(Bc, N - j * Bc);
        int row_idx = bx * Br + tx;
        int col_idx = j * Bc;
        int max_c = min(cur_Bc, row_idx - col_idx + 1);

        // load Kj,Vj to SRAM: coalesced access
        for (int x = 0; x < cur_Bc * Dim; x += WARP_SIZE) {
            if (x + tx < cur_Bc * Dim) {
                Kj_half[x + tx] = K[qkv_offset + j * tile_size + x + tx];
                Vj_half[x + tx] = V[qkv_offset + j * tile_size + x + tx];
            }
        }
        __syncthreads();

        // record l,m
        float prev_row_m = row_m;
        float prev_row_l = row_l;

        // compute Sij = Qi * Kj^T
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> mat_q;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> mat_k;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> mat_s;

        for (int x = 0; x < Br; x += 16) {
            for (int y = 0; y < Bc; y += 16) {
                wmma::fill_fragment(mat_s, 0.0f);
                for (int k = 0; k < Dim; k += 16) {
                    wmma::load_matrix_sync(mat_q, Qi_half + x * Dim + k, Dim);
                    wmma::load_matrix_sync(mat_k, Kj_half + y * Dim + k, Dim);
                    wmma::mma_sync(mat_s, mat_q, mat_k, mat_s);
                }

                // store to Sij
                wmma::store_matrix_sync(Sij + x * Bc + y, mat_s, Bc, wmma::mem_row_major);
            }
        }

        // row_m = max(S), row_l = rowsum(exp(S - row_m))
        float new_row_m = -INFINITY;
        float new_row_l = 0;
        if (tx < cur_Br) {
            for (int c = 0; c < max_c; ++c) {
                Sij[tx * Bc + c] = Sij[tx * Bc + c] * softmax_scale;
                new_row_m = max(new_row_m, Sij[tx * Bc + c]);
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            for (int c = 0; c < max_c; ++c) {
                Sij[tx * Bc + c] = __expf(Sij[tx * Bc + c] - new_row_m);
                new_row_l += Sij[tx * Bc + c];
            }

            // prepare input
            for (int c = 0; c < Bc; ++c) {
                if (c < max_c) {
                    Sij_half[tx * Bc + c] = __float2half(Sij[tx * Bc + c]);
                } else {  // zero out
                    Sij_half[tx * Bc + c] = 0;
                }
            }
        }

        // compute Sij * Vj and store to Kj
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> mat_s2;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> mat_v;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> mat_o;

        for (int x = 0; x < Br; x += 16) {
            for (int y = 0; y < Dim; y += 16) {
                wmma::fill_fragment(mat_o, 0.0f);
                for (int k = 0; k < Bc; k += 16) {
                    wmma::load_matrix_sync(mat_s2, Sij_half + x * Bc + k, Bc);
                    wmma::load_matrix_sync(mat_v, Vj_half + k * Dim + y, Dim);
                    wmma::mma_sync(mat_o, mat_s2, mat_v, mat_o);
                }
                wmma::store_matrix_sync(_Oi + x * Dim + y, mat_o, Dim, wmma::mem_row_major);
            }
        }

        // update Oi
        if (tx < cur_Br) {
            row_m = max(prev_row_m, new_row_m);

            float var1 = __expf(prev_row_m - row_m) * prev_row_l;
            float var2 = __expf(new_row_m - row_m);
            row_l = var1 + var2 * new_row_l;

            float row_l_inv = 1 / row_l;
            for (int x = 0; x < Dim; ++x) {
                Oi[tx * Dim + x] = row_l_inv * (var1 * Oi[tx * Dim + x] + var2 * _Oi[tx * Dim + x]);
            }
        }
        __syncthreads();
    }

    // write Oi to global memory
    for (int x = 0; x < cur_Br * Dim; x += WARP_SIZE) {
        if (x + tx < cur_Br * Dim) {
            O[qkv_offset + bx * tile_size + x + tx] = Oi[x + tx];
        }
    }
}

__global__ void permute_tf32_kernel(float* q, float* k, float* v, const float* inp,
                                    int B, int N, int NH, int d) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t b = idx / (N * NH * d);
    size_t n = (idx % (N * NH * d)) / (NH * d);
    size_t nh = (idx % (NH * d)) / d;
    size_t d_ = idx % d;

    if (b < B && n < N && nh < NH) {
        size_t inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (nh * d) + d_;
        size_t qkv_idx = b * NH * N * d + nh * N * d + n * d + d_;

        q[qkv_idx] = wmma::__float_to_tf32(inp[inp_idx]);
        k[qkv_idx] = wmma::__float_to_tf32(inp[inp_idx + NH * d]);
        v[qkv_idx] = wmma::__float_to_tf32(inp[inp_idx + 2 * NH * d]);
    }
}

__global__ void permute_half_kernel(half* q, half* k, half* v, const float* inp,
                                    int B, int N, int NH, int d) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t b = idx / (N * NH * d);
    size_t n = (idx % (N * NH * d)) / (NH * d);
    size_t nh = (idx % (NH * d)) / d;
    size_t d_ = idx % d;

    if (b < B && n < N && nh < NH) {
        size_t inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (nh * d) + d_;
        size_t qkv_idx = b * NH * N * d + nh * N * d + n * d + d_;

        q[qkv_idx] = __float2half(inp[inp_idx]);
        k[qkv_idx] = __float2half(inp[inp_idx + NH * d]);
        v[qkv_idx] = __float2half(inp[inp_idx + 2 * NH * d]);
    }
}

void print_sram_size_per_block() {
    int sram_size = 3 * Br * Dim * sizeof(float) + Bc * Br * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d; Requested shared memory: %d \n", max_sram_size, sram_size);
}

void flashattention2_forward(float* out, float* inp, float* qkvr, float* workspace,
                             int B, int T, int C, int NH) {

    int HC = C / NH;

    const int Tr = ceil((float)T / Br), Tc = ceil((float)T / Bc);
    const float softmax_scale = 1.0 / sqrt(HC);

    // permute
    int permute_threads = B * NH * T * HC;
    permute_tf32_kernel<<<ceil((float)permute_threads / BLOCK_SIZE), BLOCK_SIZE>>>(qkvr, qkvr + B * NH * T * HC, qkvr + 2 * B * NH * T * HC, inp, B, T, NH, HC);
    // permute_half_kernel<<<ceil((float)permute_threads / BLOCK_SIZE), BLOCK_SIZE>>>((half *)qkvr, (half *)qkvr + B * NH * T * HC, (half *)qkvr + 2 * B * NH * T * HC, inp, B, T, NH, HC);

    // forward pass
    dim3 grid_dim(Tr, B, NH);
    dim3 block_dim(WARP_SIZE, 1, 1);
    flashattn2_tf32_kernel<<<grid_dim, block_dim>>>(workspace, qkvr, qkvr + B * NH * T * HC, qkvr + 2 * B * NH * T * HC, T, Tr, Tc, softmax_scale);
    // flashattn2_half_kernel<<<grid_dim, block_dim>>>(workspace, (half *)qkvr, (half *)qkvr + B * NH * T * HC, (half *)qkvr + 2 * B * NH * T * HC, T, Tr, Tc, softmax_scale);

    // unpermute
    int unpermute_threads = B * NH * T * HC;
    unpermute_kernel<<<ceil((float)unpermute_threads / BLOCK_SIZE), BLOCK_SIZE>>>(workspace, out, B, T, NH, HC);
}

#endif