#include "../include/utils.h"

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <string>
#include <fstream>

// Utility class for dual output to console and file
class DualOutput {
 public:
  explicit DualOutput(const std::string& filename) : file_(filename) {
    if (!file_.is_open()) {
      std::cerr << "Warning: Could not open " << filename << " for writing." << std::endl;
    }
  }
  
  template<typename T>
  DualOutput& operator<<(const T& value) {
    std::cout << value;
    if (file_.is_open()) {
      file_ << value;
      file_.flush();  // Ensure immediate writing
    }
    return *this;
  }
  
  // Handle std::endl and other manipulators
  DualOutput& operator<<(std::ostream& (*manip)(std::ostream&)) {
    std::cout << manip;
    if (file_.is_open()) {
      file_ << manip;
      file_.flush();
    }
    return *this;
  }
  
 private:
  std::ofstream file_;
};

void PrintUsage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
  std::cout << "Options:\n";
  std::cout << "  -h, --help           Show this help message\n";
  std::cout << "  -s, --sizes SIZES    Comma-separated matrix sizes (default: 64,128,256,512,1024)\n";
  std::cout << "  -o, --output FILE    Output CSV file (default: results/benchmark_results.csv)\n";
  std::cout << "  -l, --log FILE       Log all output to file (default: no logging)\n";
  std::cout << "  -v, --verbose        Enable verbose output with matrix printing\n";
  std::cout << "  --cpu-only           Run only CPU implementation\n";
  std::cout << "  --gpu-only           Run only GPU implementations\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << program_name << " -s 256,512,1024\n";
  std::cout << "  " << program_name << " --verbose --output my_results.csv\n";
  std::cout << "  " << program_name << " --cpu-only -s 64,128 --log output.txt\n";
}

std::vector<int> ParseSizes(const std::string& sizes_str) {
  std::vector<int> sizes;
  size_t start = 0;
  size_t end = sizes_str.find(',');
  
  while (end != std::string::npos) {
    sizes.push_back(std::stoi(sizes_str.substr(start, end - start)));
    start = end + 1;
    end = sizes_str.find(',', start);
  }
  sizes.push_back(std::stoi(sizes_str.substr(start)));
  
  return sizes;
}

int main(int argc, char* argv[]) {
  // Default configuration
  std::vector<int> test_sizes = {64, 128, 256, 512, 1024};
  std::string output_file = "results/benchmark_results.csv";
  std::string log_file = "";
  bool verbose = false;
  bool cpu_only = false;
  bool gpu_only = false;
  
  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      PrintUsage(argv[0]);
      return 0;
    } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sizes") == 0) {
      if (i + 1 < argc) {
        test_sizes = ParseSizes(argv[++i]);
      } else {
        std::cerr << "Error: --sizes requires an argument\n";
        return 1;
      }
    } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
      if (i + 1 < argc) {
        output_file = argv[++i];
      } else {
        std::cerr << "Error: --output requires an argument\n";
        return 1;
      }
    } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--log") == 0) {
      if (i + 1 < argc) {
        log_file = argv[++i];
      } else {
        std::cerr << "Error: --log requires an argument\n";
        return 1;
      }
    } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
      verbose = true;
    } else if (strcmp(argv[i], "--cpu-only") == 0) {
      cpu_only = true;
    } else if (strcmp(argv[i], "--gpu-only") == 0) {
      gpu_only = true;
    } else {
      std::cerr << "Error: Unknown option " << argv[i] << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }
  
  if (cpu_only && gpu_only) {
    std::cerr << "Error: Cannot specify both --cpu-only and --gpu-only\n";
    return 1;
  }
  
  // Create dual output object if log file specified
  std::unique_ptr<DualOutput> out;
  if (!log_file.empty()) {
    out = std::make_unique<DualOutput>(log_file);
    *out << "=== CUDA Matrix Multiplication Benchmark ===" << std::endl;
    if (cpu_only) {
      *out << "Running CPU implementation only" << std::endl;
    } else if (gpu_only) {
      *out << "Running GPU implementations only" << std::endl;
    } else {
      *out << "Comparing CPU vs GPU (Naive) vs GPU (Shared Memory) implementations" << std::endl;
    }
  } else {
    std::cout << "=== CUDA Matrix Multiplication Benchmark ===" << std::endl;
    if (cpu_only) {
      std::cout << "Running CPU implementation only" << std::endl;
    } else if (gpu_only) {
      std::cout << "Running GPU implementations only" << std::endl;
    } else {
      std::cout << "Comparing CPU vs GPU (Naive) vs GPU (Shared Memory) implementations" << std::endl;
    }
  }
    std::cout << "Running GPU implementations only" << std::endl;
  }
  
  // Storage for timing results
  std::vector<double> cpu_times, gpu_naive_times, gpu_shared_times;
  
  Timer timer;
  
  for (int matrix_size : test_sizes) {
    std::cout << "\n--- Testing " << matrix_size << "x" << matrix_size 
              << " matrices ---" << std::endl;
    
    // Calculate memory requirements
    size_t matrix_memory = matrix_size * matrix_size * sizeof(float);
    size_t total_memory = 3 * matrix_memory;  // A, B, and C matrices
    
    std::cout << "Memory required: " 
              << (total_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    // Allocate host memory
    std::unique_ptr<float[]> matrix_a(new float[matrix_size * matrix_size]);
    std::unique_ptr<float[]> matrix_b(new float[matrix_size * matrix_size]);
    std::unique_ptr<float[]> result_cpu(new float[matrix_size * matrix_size]);
    std::unique_ptr<float[]> result_gpu_naive(new float[matrix_size * matrix_size]);
    std::unique_ptr<float[]> result_gpu_shared(new float[matrix_size * matrix_size]);
    
    // Initialize matrices with random values
    std::cout << "Initializing matrices..." << std::endl;
    InitializeMatrix(matrix_a.get(), matrix_size, matrix_size, 0.0f, 1.0f);
    InitializeMatrix(matrix_b.get(), matrix_size, matrix_size, 0.0f, 1.0f);
    
    // Print small matrices for verification if verbose mode
    if (verbose && matrix_size <= 8) {
      PrintMatrix(matrix_a.get(), matrix_size, matrix_size, "Matrix A");
      PrintMatrix(matrix_b.get(), matrix_size, matrix_size, "Matrix B");
    }
    
    // CPU Implementation
    if (!gpu_only) {
      std::cout << "Running CPU implementation..." << std::endl;
      timer.Start();
      CpuMatrixMultiply(matrix_a.get(), matrix_b.get(), result_cpu.get(), matrix_size);
      timer.Stop();
      double cpu_time = timer.ElapsedMs();
      cpu_times.push_back(cpu_time);
      std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
      
      if (verbose && matrix_size <= 8) {
        PrintMatrix(result_cpu.get(), matrix_size, matrix_size, "CPU Result");
      }
    }
    
    // GPU Naive Implementation
    if (!cpu_only) {
      std::cout << "Running GPU naive implementation..." << std::endl;
      timer.Start();
      GpuNaiveMatrixMultiply(matrix_a.get(), matrix_b.get(), 
                            result_gpu_naive.get(), matrix_size);
      timer.Stop();
      double gpu_naive_time = timer.ElapsedMs();
      gpu_naive_times.push_back(gpu_naive_time);
      std::cout << "GPU naive time: " << gpu_naive_time << " ms" << std::endl;
      
      // Verify GPU naive result (only if CPU was also run)
      if (!gpu_only) {
        bool naive_correct = VerifyResult(result_cpu.get(), result_gpu_naive.get(), 
                                         matrix_size * matrix_size);
        std::cout << "GPU naive verification: " 
                  << (naive_correct ? "PASSED" : "FAILED") << std::endl;
        if (!naive_correct) {
          std::cout << "ERROR: GPU naive implementation produced incorrect results!" 
                    << std::endl;
          return 1;
        }
      }
      
      if (verbose && matrix_size <= 8) {
        PrintMatrix(result_gpu_naive.get(), matrix_size, matrix_size, 
                   "GPU Naive Result");
      }
    }
    
    // GPU Shared Memory Implementation
    if (!cpu_only) {
      std::cout << "Running GPU shared memory implementation..." << std::endl;
      timer.Start();
      GpuSharedMatrixMultiply(matrix_a.get(), matrix_b.get(), 
                             result_gpu_shared.get(), matrix_size);
      timer.Stop();
      double gpu_shared_time = timer.ElapsedMs();
      gpu_shared_times.push_back(gpu_shared_time);
      std::cout << "GPU shared memory time: " << gpu_shared_time << " ms" << std::endl;
      
      // Verify GPU shared memory result (only if CPU was also run)
      if (!gpu_only) {
        bool shared_correct = VerifyResult(result_cpu.get(), result_gpu_shared.get(), 
                                          matrix_size * matrix_size);
        std::cout << "GPU shared memory verification: " 
                  << (shared_correct ? "PASSED" : "FAILED") << std::endl;
        if (!shared_correct) {
          std::cout << "ERROR: GPU shared memory implementation produced incorrect results!" 
                    << std::endl;
          return 1;
        }
      }
      
      if (verbose && matrix_size <= 8) {
        PrintMatrix(result_gpu_shared.get(), matrix_size, matrix_size, 
                   "GPU Shared Result");
      }
    }
    
    // Calculate and display speedups (only if both CPU and GPU were run)
    if (!cpu_only && !gpu_only) {
      double cpu_time = cpu_times.back();
      double gpu_naive_time = gpu_naive_times.back();
      double gpu_shared_time = gpu_shared_times.back();
      
      double naive_speedup = cpu_time / gpu_naive_time;
      double shared_speedup = cpu_time / gpu_shared_time;
      double shared_vs_naive = gpu_naive_time / gpu_shared_time;
      
      std::cout << "Performance Summary:" << std::endl;
      std::cout << "  CPU vs GPU Naive speedup: " << naive_speedup << "x" << std::endl;
      std::cout << "  CPU vs GPU Shared speedup: " << shared_speedup << "x" << std::endl;
      std::cout << "  GPU Shared vs Naive speedup: " << shared_vs_naive << "x" << std::endl;
    } else if (!cpu_only) {
      // Compare GPU implementations if both were run
      if (gpu_naive_times.size() > 0 && gpu_shared_times.size() > 0) {
        double gpu_naive_time = gpu_naive_times.back();
        double gpu_shared_time = gpu_shared_times.back();
        double shared_vs_naive = gpu_naive_time / gpu_shared_time;
        std::cout << "GPU Shared vs Naive speedup: " << shared_vs_naive << "x" << std::endl;
      }
    }
  }
  
  // Save results to CSV file
  if (!cpu_only && !gpu_only) {
    SaveResultsToFile(output_file, test_sizes, cpu_times, gpu_naive_times, gpu_shared_times);
  }
  
  // Print final summary
  if (!cpu_only && !gpu_only) {
    std::cout << "\n=== Final Performance Summary ===" << std::endl;
    std::cout << "Matrix Size | CPU (ms) | GPU Naive (ms) | GPU Shared (ms) | Naive Speedup | Shared Speedup" << std::endl;
    std::cout << "------------|----------|----------------|-----------------|---------------|---------------" << std::endl;
    
    for (size_t i = 0; i < test_sizes.size(); ++i) {
      double naive_speedup = cpu_times[i] / gpu_naive_times[i];
      double shared_speedup = cpu_times[i] / gpu_shared_times[i];
      
      std::cout << std::setw(11) << test_sizes[i] << " | "
                << std::setw(8) << std::setprecision(2) << std::fixed << cpu_times[i] << " | "
                << std::setw(14) << gpu_naive_times[i] << " | "
                << std::setw(15) << gpu_shared_times[i] << " | "
                << std::setw(13) << naive_speedup << " | "
                << std::setw(14) << shared_speedup << std::endl;
    }
  }
  
  std::cout << "\nBenchmark completed successfully!" << std::endl;
  return 0;
}
