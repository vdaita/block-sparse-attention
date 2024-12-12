#include <cuda_runtime.h>
#include "flash_decoding.cuh"
#include <cmath>
#include <random>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>

// GPT
void checkLastCudaError() {
    cudaError_t error = cudaGetLastError(); // Get the last error
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error)); // Print the error string
    } else {
        printf("No CUDA error.\n");
    }
}


// Claude
float getNextFloat() {
    float value;
    if (scanf("%f", &value) == 1) {
        return value;
    }
    // Handle error or end of input
    return 0.0; // Or use a special value to indicate error
}

int main(int argc, char** argv){
    int B = 1;
    int T = 64;

    float* q = (float*) malloc(B * D * sizeof(float));
    float* k = (float*) malloc(B * T * D * sizeof(float));
    float* v = (float*) malloc(B * T * D * sizeof(float));

    float* device_q; 
    cudaMalloc((void**) &device_q, B * D * sizeof(float));
    float* device_k; 
    cudaMalloc((void**) &device_k, B * T * D * sizeof(float));
    float* device_v; 
    cudaMalloc((void**) &device_v, B * T * D * sizeof(float));

    for(int i = 0; i < B * D; i++){
        q[i] = getNextFloat();
    }

    for(int i = 0; i < B * T * D; i++){
        k[i] = getNextFloat();
    }

    for(int i = 0; i < B * T * D; i++){
        v[i] = getNextFloat();
    }

    float* target_output = (float*) malloc(B * D * sizeof(float));
    for(int i = 0; i < B * D; i++){
        target_output[i] = getNextFloat();
    }

    // std::mt19937 gen(42);
    // std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // for(int i = 0; i < B * D; i++){
    //     q[i] = dist(gen);
    // }

    // for(int i = 0; i < B * T * D; i++){
    //     k[i] = dist(gen);
    //     v[i] = dist(gen);
    // }

    cudaMemcpy(device_q, q, B * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, k, B * T * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, v, B * T * D * sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks_per_head = min((T + BLOCK_TOKENS - 1) / BLOCK_TOKENS, 1);
    dim3 gridDim(1, num_blocks_per_head, B);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_TOKENS, 1);

    float* device_o;
    cudaMalloc((void**) &device_o, B * D * num_blocks_per_head * sizeof(float)); 
    float* o = (float*) malloc(B * D * num_blocks_per_head * sizeof(float));

    float* device_o_sum;
    cudaMalloc((void**) &device_o_sum, B * num_blocks_per_head * sizeof(float));
    float* device_o_max;
    cudaMalloc((void**) &device_o_max, B * num_blocks_per_head * sizeof(float));

    float* o_sum = (float*) malloc(B * num_blocks_per_head * sizeof(float));
    float* o_max = (float*) malloc(B * num_blocks_per_head * sizeof(float));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Shared memory per block: %d bytes\n", prop.sharedMemPerBlock);
    printf("Num blocks per head %d\n", num_blocks_per_head);

    shared_split_k_kernel<<<gridDim, blockDim>>>(
        device_q,
        device_k,
        device_v,
        device_o,
        device_o_sum,
        device_o_max,
        B,
        T
    );

    checkLastCudaError();

    cudaMemcpy(o, device_o, B * D * num_blocks_per_head * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(o_sum, device_o_sum, B * num_blocks_per_head * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(o_max, device_o_max, B * num_blocks_per_head * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < D; i++){
        printf("%f ", (o[i] / o_sum[0]));
    }
    printf("\n");

    bool works = true;
    for(int i = 0; i < D; i++){
        if(abs((o[i] / o_sum[0]) - target_output[i]) > 0.02){
            works = false;
            break;
        }
    }

    if(works){
        printf("Works!\n");
    } else {
        printf("Doesn't work!\n");
    }
}