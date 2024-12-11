#include <cuda_runtime.h>
#include <torch/extension.h>

#define D 128
#define BLOCK_SIZE 32 // perform some type of calculation
#define NUM_CHUNKS 32

__global__
void shared_split_k_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    const int B,
    const int T,
    const int num_to_process
) {
    int tx = threadIdx.x; // represents the hidden dim dimension
    int ty = threadIdx.y; // represents the thread id dimension

    int block_width = blockDim.x;
    int block_height = blockDim.y;

    int bx = blockIdx.x; // presents the index dimension

    int batch = blockIdx.z;

    // load data into shared memory
    __shared__ float shared_q[D];

    // perform reduction in shared memory
    for(int i = ty * block_width + tx; i < D; i += block_width * block_height){
        shared_q[i] = Q[batch * D + i];
    }

    // write to global memory
}