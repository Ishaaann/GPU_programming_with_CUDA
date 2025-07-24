#ifndef CAPSTONE_INCLUDE_UTILS_H_
#define CAPSTONE_INCLUDE_UTILS_H_

#include <chrono>  // NOLINT(build/c++11)
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Timer class for performance measurement
class Timer {
 public:
  void Start() {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  
  void Stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
  }
  
  double ElapsedMs() const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time_ - start_time_);
    return duration.count() / 1000.0;  // Convert to milliseconds
  }

 private:
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point end_time_;
};
// Matrix utility functions
void InitializeMatrix(float* matrix, int rows, int cols, 
                     float min_val = 0.0f, float max_val = 1.0f);
void PrintMatrix(const float* matrix, int rows, int cols, 
                const std::string& name = "Matrix");
bool VerifyResult(const float* cpu_result, const float* gpu_result, 
                 int size, float tolerance = 1e-3f);
void SaveResultsToFile(const std::string& filename, 
                      const std::vector<int>& sizes,
                      const std::vector<double>& cpu_times,
                      const std::vector<double>& gpu_naive_times,
                      const std::vector<double>& gpu_shared_times);

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Function declarations for different implementations
void CpuMatrixMultiply(const float* matrix_a, const float* matrix_b, 
                      float* result, int matrix_size);
void GpuNaiveMatrixMultiply(const float* matrix_a, const float* matrix_b, 
                           float* result, int matrix_size);
void GpuSharedMatrixMultiply(const float* matrix_a, const float* matrix_b, 
                            float* result, int matrix_size);

#endif  // CAPSTONE_INCLUDE_UTILS_H_
