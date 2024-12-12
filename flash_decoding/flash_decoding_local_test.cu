#include <cuda_runtime.h>
#include "flash_decoding.cuh"
#include <cmath>
#include <random>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>

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
    int T = 32;

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

    float* device_o;
    cudaMalloc((void**) &device_o, B * D * sizeof(float)); 
    float* o = (float*) malloc(B * D * sizeof(float));

    int num_blocks_for_head = min((T + BLOCK_TOKENS - 1) / BLOCK_TOKENS, 8);
    dim3 gridDim(1, num_blocks_for_head, B);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_TOKENS, 1);

    shared_split_k_kernel<<<gridDim, blockDim>>>(
        device_q,
        device_k,
        device_v,
        device_o,
        B,
        T
    );

    cudaMemcpy(o, device_o, B * D * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < D; i++){
        printf("%f ", o[i]);
    }
    printf("\n");
}