#!/usr/bin/env python3
"""
Project verification script for CUDA Matrix Multiplication Benchmark
Checks that all required files exist and have proper content structure.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and return its status."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {description}: Found ({size} bytes)")
        return True
    else:
        print(f"✗ {description}: Missing")
        return False

def check_directory_structure():
    """Verify the expected directory structure."""
    print("=== Directory Structure Check ===")
    
    required_dirs = [
        ("include", "Header files directory"),
        ("src", "Source files directory"),
        ("results", "Results output directory")
    ]
    
    all_dirs_exist = True
    for dir_name, description in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {description}: {dir_name}/")
        else:
            print(f"✗ {description}: {dir_name}/ (missing)")
            all_dirs_exist = False
    
    return all_dirs_exist

def check_source_files():
    """Verify all required source files exist."""
    print("\n=== Source Files Check ===")
    
    required_files = [
        ("include/utils.h", "Header file with declarations"),
        ("src/main.cu", "Main program with benchmarking"),
        ("src/cpu_matrix_mul.cpp", "CPU implementation"),
        ("src/gpu_naive.cu", "Naive GPU implementation"),
        ("src/gpu_shared.cu", "Shared memory GPU implementation"),
        ("Makefile", "Build configuration"),
        ("README.md", "Project documentation")
    ]
    
    all_files_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    return all_files_exist

def check_file_content():
    """Check for key content in critical files."""
    print("\n=== Content Verification ===")
    
    checks_passed = True
    
    # Check utils.h for key declarations
    try:
        with open("include/utils.h", "r") as f:
            content = f.read()
            if "CUDA_CHECK" in content and "Timer" in content:
                print("✓ utils.h: Contains required macros and classes")
            else:
                print("✗ utils.h: Missing key declarations")
                checks_passed = False
    except FileNotFoundError:
        print("✗ utils.h: File not found")
        checks_passed = False
    
    # Check for CUDA kernels
    cuda_files = ["src/gpu_naive.cu", "src/gpu_shared.cu"]
    for cuda_file in cuda_files:
        try:
            with open(cuda_file, "r") as f:
                content = f.read()
                if "__global__" in content and "<<<" in content:
                    print(f"✓ {cuda_file}: Contains CUDA kernel")
                else:
                    print(f"✗ {cuda_file}: Missing CUDA kernel syntax")
                    checks_passed = False
        except FileNotFoundError:
            print(f"✗ {cuda_file}: File not found")
            checks_passed = False
    
    # Check Makefile for nvcc
    try:
        with open("Makefile", "r") as f:
            content = f.read()
            if "nvcc" in content.lower() and "cuda" in content.lower():
                print("✓ Makefile: Contains CUDA build configuration")
            else:
                print("✗ Makefile: Missing CUDA configuration")
                checks_passed = False
    except FileNotFoundError:
        print("✗ Makefile: File not found")
        checks_passed = False
    
    return checks_passed

def check_cuda_environment():
    """Check if CUDA environment is available."""
    print("\n=== CUDA Environment Check ===")
    
    # Check for nvcc compiler
    nvcc_available = os.system("which nvcc > /dev/null 2>&1") == 0
    if nvcc_available:
        print("✓ nvcc compiler: Available")
        # Try to get CUDA version
        os.system("nvcc --version | head -n 4")
    else:
        print("✗ nvcc compiler: Not found in PATH")
        return False
    
    # Check for CUDA runtime
    try:
        os.system("nvidia-smi > /dev/null 2>&1")
        print("✓ NVIDIA GPU: Detected")
    except:
        print("? NVIDIA GPU: Cannot verify (nvidia-smi not available)")
    
    return nvcc_available

def main():
    """Main verification function."""
    print("CUDA Matrix Multiplication Project Verification")
    print("=" * 50)
    
    # Change to project directory if script is run from elsewhere
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run all checks
    dir_check = check_directory_structure()
    file_check = check_source_files()
    content_check = check_file_content()
    cuda_check = check_cuda_environment()
    
    # Summary
    print("\n=== Verification Summary ===")
    
    all_passed = dir_check and file_check and content_check and cuda_check
    
    if all_passed:
        print("🎉 All checks passed! Project is ready to build and run.")
        print("\nNext steps:")
        print("  1. Run 'make clean' to clean any previous builds")
        print("  2. Run 'make' to build the project")
        print("  3. Run './matrix_benchmark' to execute the benchmark")
        print("  4. Check 'results/benchmark_results.csv' for detailed results")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        if not cuda_check:
            print("\nNote: If CUDA is not available, you can still review the code structure.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
