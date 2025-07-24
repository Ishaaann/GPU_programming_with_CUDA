#!/bin/bash

# Script to demonstrate output redirection for CUDA Matrix Multiplication Benchmark
# This script shows different ways to capture program output

echo "=== CUDA Matrix Multiplication Output Capture Demo ==="
echo

# Method 1: Display on console AND save to file using tee
echo "Method 1: Using 'tee' to display and save output..."
echo "./matrix_benchmark -s 64,128 | tee output.txt"
echo "This will show output on screen AND save to output.txt"
echo

# Method 2: Save to file only (no console output)
echo "Method 2: Redirect to file only..."
echo "./matrix_benchmark -s 256 > output_only.txt 2>&1"
echo "This will save ALL output (stdout and stderr) to output_only.txt"
echo

# Method 3: Append to existing file
echo "Method 3: Append to existing file..."
echo "./matrix_benchmark -s 512 >> output.txt 2>&1"
echo "This will APPEND output to existing output.txt"
echo

# Method 4: Separate stdout and stderr
echo "Method 4: Separate stdout and stderr..."
echo "./matrix_benchmark -s 64 > results.txt 2> errors.txt"
echo "This will save normal output to results.txt and errors to errors.txt"
echo

echo "Usage examples:"
echo "  bash $0 demo     # Run demonstration"
echo "  bash $0 test     # Run actual tests with output capture"

if [ "$1" = "demo" ]; then
    echo
    echo "Running demonstration with tee (Method 1)..."
    ./matrix_benchmark -s 64 | tee demo_output.txt
    echo
    echo "Output saved to demo_output.txt"
    echo "File size: $(wc -c < demo_output.txt) bytes"
    
elif [ "$1" = "test" ]; then
    echo
    echo "Running comprehensive test with output capture..."
    
    # Test 1: Small matrices with verbose output
    echo "Test 1: Small matrices (verbose) -> verbose_output.txt"
    ./matrix_benchmark -v -s 64,128 | tee verbose_output.txt
    
    # Test 2: GPU-only test
    echo "Test 2: GPU-only test -> gpu_only_output.txt"
    ./matrix_benchmark --gpu-only -s 256,512 > gpu_only_output.txt 2>&1
    
    # Test 3: CPU-only test
    echo "Test 3: CPU-only test -> cpu_only_output.txt"
    ./matrix_benchmark --cpu-only -s 64,128,256 > cpu_only_output.txt 2>&1
    
    echo
    echo "All tests completed. Output files created:"
    ls -la *_output.txt
    
else
    echo
    echo "Run with 'demo' or 'test' argument to see examples in action."
fi
