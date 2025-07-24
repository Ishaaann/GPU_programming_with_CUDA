# Compiler settings
NVCC = nvcc
CXX = g++

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_50 -std=c++11
CXX_FLAGS = -O3 -std=c++11

# Directories
SRC_DIR = src
INCLUDE_DIR = include
RESULTS_DIR = results

# Source files
CUDA_SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/gpu_naive.cu $(SRC_DIR)/gpu_shared.cu
CPP_SOURCES = $(SRC_DIR)/cpu_matrix_mul.cpp

# Output executable
TARGET = matrix_benchmark

# Default target
all: $(TARGET)

# Create results directory
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

# Build the main executable
$(TARGET): $(CUDA_SOURCES) $(CPP_SOURCES) | $(RESULTS_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) $(CUDA_SOURCES) $(CPP_SOURCES) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)
	rm -rf $(RESULTS_DIR)

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Build and run
test: clean all run

# Debug build
debug: NVCC_FLAGS += -g -G
debug: $(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the project (default)"
	@echo "  clean   - Remove built files"
	@echo "  run     - Run the benchmark"
	@echo "  test    - Clean, build, and run"
	@echo "  debug   - Build with debug information"
	@echo "  help    - Show this help message"

.PHONY: all clean run test debug help
