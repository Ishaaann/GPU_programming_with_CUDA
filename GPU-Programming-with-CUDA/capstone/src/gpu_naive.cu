#include "../include/utils.h"
#include <cuda_runtime.h>

// Naive GPU matrix multiplication kernel
__global__ void NaiveMatrixMulKernel(const float* matrix_a, const float* matrix_b, 
                                     float* result, int matrix_size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < matrix_size && col < matrix_size) {
    float sum = 0.0f;
    for (int k = 0; k < matrix_size; ++k) {
      sum += matrix_a[row * matrix_size + k] * matrix_b[k * matrix_size + col];
    }
    result[row * matrix_size + col] = sum;
  }
}

void GpuNaiveMatrixMultiply(const float* matrix_a, const float* matrix_b, 
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
  NaiveMatrixMulKernel<<<grid_dim, block_dim>>>(d_matrix_a, d_matrix_b, 
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
