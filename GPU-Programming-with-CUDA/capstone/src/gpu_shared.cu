#include "../include/utils.h"
#include <cuda_runtime.h>

// Shared memory matrix multiplication kernel
__global__ void SharedMatrixMulKernel(const float* matrix_a, const float* matrix_b, 
                                      float* result, int matrix_size) {
  const int kBlockSize = 16;
  
  // Shared memory for tile loading
  __shared__ float tile_a[kBlockSize][kBlockSize];
  __shared__ float tile_b[kBlockSize][kBlockSize];
  
  int row = blockIdx.y * kBlockSize + threadIdx.y;
  int col = blockIdx.x * kBlockSize + threadIdx.x;
  
  float sum = 0.0f;
  
  // Loop over tiles
  for (int tile = 0; tile < (matrix_size + kBlockSize - 1) / kBlockSize; ++tile) {
    // Load tile into shared memory
    int tile_row = tile * kBlockSize + threadIdx.x;
    int tile_col = tile * kBlockSize + threadIdx.y;
    
    // Load A tile
    if (row < matrix_size && tile_row < matrix_size) {
      tile_a[threadIdx.y][threadIdx.x] = matrix_a[row * matrix_size + tile_row];
    } else {
      tile_a[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    // Load B tile
    if (col < matrix_size && tile_col < matrix_size) {
      tile_b[threadIdx.y][threadIdx.x] = matrix_b[tile_col * matrix_size + col];
    } else {
      tile_b[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    // Synchronize to ensure tiles are loaded
    __syncthreads();
    
    // Compute partial sum for this tile
    for (int k = 0; k < kBlockSize; ++k) {
      sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }
    
    // Synchronize before loading next tile
    __syncthreads();
  }
  
  // Write result
  if (row < matrix_size && col < matrix_size) {
    result[row * matrix_size + col] = sum;
  }
}

void GpuSharedMatrixMultiply(const float* matrix_a, const float* matrix_b, 
                            float* result, int matrix_size) {
  // Device memory pointers
  float *d_matrix_a, *d_matrix_b, *d_result;
  
  // Calculate memory size
  size_t size = matrix_size * matrix_size * sizeof(float);
  
  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_matrix_a, size));
  CUDA_CHECK(cudaMalloc(&d_matrix_b, size));
  CUDA_CHECK(cudaMalloc(&d_result, size));
  
  // Copy input matrices to device
  CUDA_CHECK(cudaMemcpy(d_matrix_a, matrix_a, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_matrix_b, matrix_b, size, cudaMemcpyHostToDevice));
  
  // Define block and grid dimensions
  const int kBlockSize = 16;
  dim3 block_dim(kBlockSize, kBlockSize);
  dim3 grid_dim((matrix_size + kBlockSize - 1) / kBlockSize, 
                (matrix_size + kBlockSize - 1) / kBlockSize);
  
  // Launch kernel
  SharedMatrixMulKernel<<<grid_dim, block_dim>>>(d_matrix_a, d_matrix_b, 
                                                 d_result, matrix_size);
  
  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());
  
  // Synchronize to ensure kernel completion
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost));
  
  // Free device memory
  cudaFree(d_matrix_a);
  cudaFree(d_matrix_b);
  cudaFree(d_result);
}
