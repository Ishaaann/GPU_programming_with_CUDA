#include "../include/utils.h"
#include <iostream>
#include <vector>
#include <memory>

int main() {
    std::cout << "=== CUDA Matrix Multiplication Benchmark ===" << std::endl;
    std::cout << "Comparing CPU vs GPU (Naive) vs GPU (Shared Memory) implementations" << std::endl;
    
    // Test different matrix sizes
    std::vector<int> test_sizes = {64, 128, 256, 512, 1024};
    
    // Storage for timing results
    std::vector<double> cpu_times, gpu_naive_times, gpu_shared_times;
    
    Timer timer;
    
    for (int N : test_sizes) {
        std::cout << "\n--- Testing " << N << "x" << N << " matrices ---" << std::endl;
        
        // Calculate memory requirements
        size_t matrix_size = N * N * sizeof(float);
        size_t total_memory = 3 * matrix_size; // A, B, and C matrices
        
        std::cout << "Memory required: " << (total_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
        
        // Allocate host memory
        std::unique_ptr<float[]> A(new float[N * N]);
        std::unique_ptr<float[]> B(new float[N * N]);
        std::unique_ptr<float[]> C_cpu(new float[N * N]);
        std::unique_ptr<float[]> C_gpu_naive(new float[N * N]);
        std::unique_ptr<float[]> C_gpu_shared(new float[N * N]);
        
        // Initialize matrices with random values
        std::cout << "Initializing matrices..." << std::endl;
        initialize_matrix(A.get(), N, N, 0.0f, 1.0f);
        initialize_matrix(B.get(), N, N, 0.0f, 1.0f);
        
        // Print small matrices for verification
        if (N <= 8) {
            print_matrix(A.get(), N, N, "Matrix A");
            print_matrix(B.get(), N, N, "Matrix B");
        }
        
        // CPU Implementation
        std::cout << "Running CPU implementation..." << std::endl;
        timer.start();
        cpu_matrix_multiply(A.get(), B.get(), C_cpu.get(), N);
        timer.stop();
        double cpu_time = timer.elapsed_ms();
        cpu_times.push_back(cpu_time);
        std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
        
        if (N <= 8) {
            print_matrix(C_cpu.get(), N, N, "CPU Result");
        }
        
        // GPU Naive Implementation
        std::cout << "Running GPU naive implementation..." << std::endl;
        timer.start();
        gpu_naive_matrix_multiply(A.get(), B.get(), C_gpu_naive.get(), N);
        timer.stop();
        double gpu_naive_time = timer.elapsed_ms();
        gpu_naive_times.push_back(gpu_naive_time);
        std::cout << "GPU naive time: " << gpu_naive_time << " ms" << std::endl;
        
        // Verify GPU naive result
        bool naive_correct = verify_result(C_cpu.get(), C_gpu_naive.get(), N * N);
        std::cout << "GPU naive verification: " << (naive_correct ? "PASSED" : "FAILED") << std::endl;
        
        if (N <= 8) {
            print_matrix(C_gpu_naive.get(), N, N, "GPU Naive Result");
        }
        
        // GPU Shared Memory Implementation
        std::cout << "Running GPU shared memory implementation..." << std::endl;
        timer.start();
        gpu_shared_matrix_multiply(A.get(), B.get(), C_gpu_shared.get(), N);
        timer.stop();
        double gpu_shared_time = timer.elapsed_ms();
        gpu_shared_times.push_back(gpu_shared_time);
        std::cout << "GPU shared memory time: " << gpu_shared_time << " ms" << std::endl;
        
        // Verify GPU shared memory result
        bool shared_correct = verify_result(C_cpu.get(), C_gpu_shared.get(), N * N);
        std::cout << "GPU shared memory verification: " << (shared_correct ? "PASSED" : "FAILED") << std::endl;
        
        if (N <= 8) {
            print_matrix(C_gpu_shared.get(), N, N, "GPU Shared Result");
        }
        
        // Calculate and display speedups
        double naive_speedup = cpu_time / gpu_naive_time;
        double shared_speedup = cpu_time / gpu_shared_time;
        double shared_vs_naive = gpu_naive_time / gpu_shared_time;
        
        std::cout << "Performance Summary:" << std::endl;
        std::cout << "  CPU vs GPU Naive speedup: " << naive_speedup << "x" << std::endl;
        std::cout << "  CPU vs GPU Shared speedup: " << shared_speedup << "x" << std::endl;
        std::cout << "  GPU Shared vs Naive speedup: " << shared_vs_naive << "x" << std::endl;
        
        if (!naive_correct || !shared_correct) {
            std::cout << "ERROR: GPU implementation produced incorrect results!" << std::endl;
            return 1;
        }
    }
    
    // Save results to CSV file
    save_results_to_file("results/benchmark_results.csv", test_sizes, cpu_times, gpu_naive_times, gpu_shared_times);
    
    // Print final summary
    std::cout << "\n=== Final Performance Summary ===" << std::endl;
    std::cout << "Matrix Size | CPU (ms) | GPU Naive (ms) | GPU Shared (ms) | Naive Speedup | Shared Speedup" << std::endl;
    std::cout << "------------|----------|----------------|-----------------|---------------|---------------" << std::endl;
    
    for (size_t i = 0; i < test_sizes.size(); i++) {
        double naive_speedup = cpu_times[i] / gpu_naive_times[i];
        double shared_speedup = cpu_times[i] / gpu_shared_times[i];
        
        std::cout << std::setw(11) << test_sizes[i] << " | "
                  << std::setw(8) << std::setprecision(2) << std::fixed << cpu_times[i] << " | "
                  << std::setw(14) << gpu_naive_times[i] << " | "
                  << std::setw(15) << gpu_shared_times[i] << " | "
                  << std::setw(13) << naive_speedup << " | "
                  << std::setw(14) << shared_speedup << std::endl;
    }
    
    std::cout << "\nBenchmark completed successfully!" << std::endl;
    return 0;
}
