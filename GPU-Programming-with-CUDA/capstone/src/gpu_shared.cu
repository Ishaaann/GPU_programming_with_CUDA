#include "../include/utils.h"
#include <cuda_runtime.h>

// Shared memory matrix multiplication kernel
__global__ void shared_matrix_mul_kernel(const float* A, const float* B, float* C, int N) {
    const int BLOCK_SIZE = 16;
    
    // Shared memory for tile loading
    __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // Load tile into shared memory
        int tile_row = tile * BLOCK_SIZE + threadIdx.x;
        int tile_col = tile * BLOCK_SIZE + threadIdx.y;
        
        // Load A tile
        if (row < N && tile_row < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + tile_row];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load B tile
        if (col < N && tile_col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[tile_col * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void gpu_shared_matrix_multiply(const float* A, const float* B, float* C, int N) {
    // Device memory pointers
    float *d_A, *d_B, *d_C;
    
    // Calculate memory size
    size_t size = N * N * sizeof(float);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    // Copy input matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    
    // Define block and grid dimensions
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    shared_matrix_mul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize to ensure kernel completion
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
