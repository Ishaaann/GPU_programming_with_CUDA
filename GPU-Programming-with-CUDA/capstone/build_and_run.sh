#!/bin/bash

# CUDA Matrix Multiplication Project - WSL Setup and Build Script
# Run this script in WSL where nvcc is installed

echo "=== CUDA Matrix Multiplication Project Setup ==="
echo "Make sure you're running this in WSL with CUDA installed"
echo

# Navigate to the project directory (adjust path as needed)
# cd /mnt/c/Users/Ishaan/repos/Practice\ Projects/gpu/GPU-Programming-with-CUDA/capstone

# Check CUDA installation
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found:"
    nvcc --version | head -n 4
else
    echo "✗ nvcc not found. Please install CUDA toolkit."
    exit 1
fi

echo

# Check for NVIDIA GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "? nvidia-smi not available"
fi

echo

# Verify project structure
echo "Verifying project files..."
python3 verify_project.py

echo

# Build the project
echo "Building the project..."
make clean
make

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo
    echo "Running the benchmark..."
    ./matrix_benchmark
else
    echo "✗ Build failed. Check the error messages above."
    exit 1
fi

echo
echo "Check the results/ directory for detailed benchmark data."
