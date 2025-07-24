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

## Building the Project

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc compiler)
- C++ compiler (g++)
- Make utility

### Build Commands
```bash
# Clean previous builds
make clean

# Build the project
make

# Build with debug information
make debug
```

### Build Targets
- `make` or `make all` - Build the project (default)
- `make clean` - Remove built files and results directory
- `make run` - Build and run with default settings
- `make test` - Clean, build, and run the benchmark
- `make debug` - Build with debug information
- `make help` - Show available build targets

## Running the Benchmark

### Basic Usage
```bash
# Run with default settings (matrix sizes: 64,128,256,512,1024)
./matrix_benchmark

# Show help and available options
./matrix_benchmark --help
```

### Command Line Options
```bash
Options:
  -h, --help           Show help message and exit
  -s, --sizes SIZES    Comma-separated matrix sizes (default: 64,128,256,512,1024)
  -o, --output FILE    Output CSV file (default: results/benchmark_results.csv)
  -v, --verbose        Enable verbose output with matrix printing for small matrices
  --cpu-only           Run only CPU implementation
  --gpu-only           Run only GPU implementations
```

### Usage Examples
```bash
# Test specific matrix sizes
./matrix_benchmark -s 256,512,1024

# Run with verbose output and custom output file
./matrix_benchmark --verbose --output my_results.csv

# Test only CPU implementation with specific sizes
./matrix_benchmark --cpu-only -s 64,128

# Test only GPU implementations
./matrix_benchmark --gpu-only -s 512,1024

# Run with all options
./matrix_benchmark -v -s 128,256,512 -o detailed_results.csv

# Save all terminal output to a file
./matrix_benchmark -s 64,128,256 | tee output.txt

# Save output to file without displaying on console (Linux/WSL)
./matrix_benchmark -s 512,1024 > output.txt 2>&1
```

### Output Redirection
The program supports standard output redirection:
- **`| tee output.txt`** - Display output on console AND save to file
- **`> output.txt`** - Save output to file only (no console display)
- **`>> output.txt`** - Append output to existing file

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

### Default Run
```
$ ./matrix_benchmark
=== CUDA Matrix Multiplication Benchmark ===
Comparing CPU vs GPU (Naive) vs GPU (Shared Memory) implementations

--- Testing 512x512 matrices ---
Memory required: 3.00 MB
Initializing matrices...
Running CPU implementation...
CPU time: 234.56 ms
Running GPU naive implementation...
GPU naive time: 12.34 ms
GPU naive verification: PASSED
Running GPU shared memory implementation...
GPU shared memory time: 4.56 ms
GPU shared memory verification: PASSED
Performance Summary:
  CPU vs GPU Naive speedup: 19.0x
  CPU vs GPU Shared speedup: 51.4x
  GPU Shared vs Naive speedup: 2.7x
```

### Custom Sizes with Verbose Output
```
$ ./matrix_benchmark -v -s 64,128
=== CUDA Matrix Multiplication Benchmark ===
Comparing CPU vs GPU (Naive) vs GPU (Shared Memory) implementations

--- Testing 64x64 matrices ---
Memory required: 0.047 MB
Initializing matrices...

Matrix A (64x64):
   0.123    0.456    0.789   ...
   0.234    0.567    0.890   ...
   ...

CPU time: 0.15 ms
GPU naive time: 1.23 ms
GPU shared memory time: 0.89 ms
Performance Summary:
  CPU vs GPU Naive speedup: 0.12x
  CPU vs GPU Shared speedup: 0.17x
  GPU Shared vs Naive speedup: 1.38x
```

### GPU-Only Mode
```
$ ./matrix_benchmark --gpu-only -s 1024
=== CUDA Matrix Multiplication Benchmark ===
Running GPU implementations only

--- Testing 1024x1024 matrices ---
Memory required: 12.00 MB
Initializing matrices...
Running GPU naive implementation...
GPU naive time: 45.67 ms
Running GPU shared memory implementation...
GPU shared memory time: 32.10 ms
GPU Shared vs Naive speedup: 1.42x
```

## Code Quality

This project follows the **Google C++ Style Guide** standards:

- Function names use PascalCase (e.g., `InitializeMatrix`, `CpuMatrixMultiply`)
- Variable names use snake_case (e.g., `matrix_size`, `gpu_naive_time`)
- Constants use kPrefix (e.g., `kBlockSize`)
- Header guards use full path (e.g., `CAPSTONE_INCLUDE_UTILS_H_`)
- Proper include ordering and formatting
- Consistent indentation and spacing
- Meaningful variable and function names

## Command Line Interface

The program accepts various command line arguments for flexible testing:

### Required Arguments
None - all arguments are optional with sensible defaults.

### Optional Arguments
- **Matrix Sizes**: Use `-s` or `--sizes` to specify custom matrix dimensions
- **Output File**: Use `-o` or `--output` to set custom CSV output location
- **Verbose Mode**: Use `-v` or `--verbose` for detailed output including matrix printing
- **Execution Mode**: Use `--cpu-only` or `--gpu-only` to test specific implementations

### Error Handling
- Invalid arguments display help message and exit with error code 1
- Conflicting options (e.g., `--cpu-only` and `--gpu-only`) are detected and reported
- Missing required argument values are validated and reported

## Educational Value

This project demonstrates:
- CUDA programming fundamentals
- Memory hierarchy optimization  
- Performance analysis and benchmarking
- GPU vs CPU computational differences
- Shared memory usage in CUDA
- Command line argument parsing
- Google C++ Style Guide compliance
- Makefile build system usage

## Build System

The project includes a comprehensive Makefile with multiple targets:

```makefile
# Available targets
make          # Build the project (default)
make clean    # Remove built files
make run      # Build and run with default settings  
make test     # Clean, build, and run
make debug    # Build with debug information
make help     # Show available targets
```

## Repository Structure

This repository includes all necessary files for building and execution:
- **Source code** in `src/` directory with proper separation of concerns
- **Headers** in `include/` directory following Google style
- **Makefile** with comprehensive build targets and documentation
- **README.md** with detailed usage instructions and examples
- **Verification script** (`verify_project.py`) for setup validation
- **Output capture script** (`capture_output.sh`) for demonstration of output redirection

### Output Capture Utilities

The repository includes `capture_output.sh` script that demonstrates various output redirection methods:

```bash
# Show demonstration of output capture methods
./capture_output.sh demo

# Run comprehensive tests with different output capture modes
./capture_output.sh test
```

This script creates multiple output files showing different capture techniques:
- `verbose_output.txt` - Verbose output with matrix printing
- `gpu_only_output.txt` - GPU-only benchmark results  
- `cpu_only_output.txt` - CPU-only benchmark results
