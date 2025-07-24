#include "../include/utils.h"
#include <random>
#include <fstream>
#include <algorithm>
#include <cmath>

void initialize_matrix(float* matrix, int rows, int cols, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

void print_matrix(const float* matrix, int rows, int cols, const std::string& name) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):\n";
    
    // Only print small matrices to avoid clutter
    if (rows <= 8 && cols <= 8) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << std::setw(8) << std::setprecision(3) << std::fixed 
                         << matrix[i * cols + j] << " ";
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "Matrix too large to display (>" << rows << "x" << cols << ")\n";
    }
    std::cout << std::endl;
}

bool verify_result(const float* cpu_result, const float* gpu_result, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (std::abs(cpu_result[i] - gpu_result[i]) > tolerance) {
            std::cout << "Verification failed at index " << i 
                     << ": CPU=" << cpu_result[i] 
                     << ", GPU=" << gpu_result[i] 
                     << ", diff=" << std::abs(cpu_result[i] - gpu_result[i]) << std::endl;
            return false;
        }
    }
    return true;
}

void save_results_to_file(const std::string& filename, 
                         const std::vector<int>& sizes,
                         const std::vector<double>& cpu_times,
                         const std::vector<double>& gpu_naive_times,
                         const std::vector<double>& gpu_shared_times) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    
    file << "Matrix_Size,CPU_Time_ms,GPU_Naive_Time_ms,GPU_Shared_Time_ms,CPU_vs_Naive_Speedup,CPU_vs_Shared_Speedup\n";
    
    for (size_t i = 0; i < sizes.size(); i++) {
        double naive_speedup = cpu_times[i] / gpu_naive_times[i];
        double shared_speedup = cpu_times[i] / gpu_shared_times[i];
        
        file << sizes[i] << ","
             << cpu_times[i] << ","
             << gpu_naive_times[i] << ","
             << gpu_shared_times[i] << ","
             << naive_speedup << ","
             << shared_speedup << "\n";
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

void cpu_matrix_multiply(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
