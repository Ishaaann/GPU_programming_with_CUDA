# CUDA Matrix Multiplication Benchmark

A comprehensive comparison of matrix multiplication implementations using CPU and GPU (CUDA) approaches.

## Project Overview

This project implements and benchmarks three different matrix multiplication approaches:

1. **CPU Implementation**: Traditional nested loops on CPU
2. **GPU Naive Implementation**: Basic CUDA kernel with global memory access
3. **GPU Shared Memory Implementation**: Optimized CUDA kernel using shared memory tiling

## Features

- Benchmarks multiple matrix sizes (64x64 to 1024x1024)
- Automatic result verification between implementations
- Performance timing and speedup calculations
- CSV output for detailed analysis
- Memory usage reporting

## File Structure

```
capstone/
├── include/
│   └── utils.h              # Header with utility functions and declarations
├── src/
│   ├── main.cu             # Main program with benchmarking logic
│   ├── cpu_matrix_mul.cpp  # CPU matrix multiplication implementation
│   ├── gpu_naive.cu        # Naive GPU implementation
│   └── gpu_shared.cu       # Shared memory GPU implementation
├── results/                # Output directory for benchmark results
├── Makefile               # Build configuration
└── README.md              # This file
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc compiler)
- C++ compiler (g++)
- Make utility

## Building the Project

```bash
make clean
make
```

## Running the Benchmark

```bash
./matrix_benchmark
```

## Output

The program will:
1. Test multiple matrix sizes automatically
2. Display timing results for each implementation
3. Verify correctness of GPU implementations
4. Show speedup calculations
5. Save detailed results to `results/benchmark_results.csv`

## Performance Characteristics

- **CPU Implementation**: O(N³) complexity, single-threaded
- **GPU Naive**: Parallel execution but high global memory latency
- **GPU Shared Memory**: Optimized memory access pattern using tiling

Expected performance improvements:
- GPU Naive: 10-50x speedup over CPU (depending on matrix size)
- GPU Shared: Additional 2-5x improvement over naive GPU implementation

## Sample Output

```
=== CUDA Matrix Multiplication Benchmark ===
--- Testing 512x512 matrices ---
Memory required: 3.00 MB
CPU time: 234.56 ms
GPU naive time: 12.34 ms
GPU shared memory time: 4.56 ms
CPU vs GPU Naive speedup: 19.0x
CPU vs GPU Shared speedup: 51.4x
GPU Shared vs Naive speedup: 2.7x
```

## Educational Value

This project demonstrates:
- CUDA programming fundamentals
- Memory hierarchy optimization
- Performance analysis and benchmarking
- GPU vs CPU computational differences
- Shared memory usage in CUDA
