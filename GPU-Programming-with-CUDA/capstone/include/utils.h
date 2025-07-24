#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

// Timer class for performance measurement
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Convert to milliseconds
    }
};

// Matrix utility functions
void initialize_matrix(float* matrix, int rows, int cols, float min_val = 0.0f, float max_val = 1.0f);
void print_matrix(const float* matrix, int rows, int cols, const std::string& name = "Matrix");
bool verify_result(const float* cpu_result, const float* gpu_result, int size, float tolerance = 1e-3f);
void save_results_to_file(const std::string& filename, 
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
void cpu_matrix_multiply(const float* A, const float* B, float* C, int N);
void gpu_naive_matrix_multiply(const float* A, const float* B, float* C, int N);
void gpu_shared_matrix_multiply(const float* A, const float* B, float* C, int N);

#endif // UTILS_H
