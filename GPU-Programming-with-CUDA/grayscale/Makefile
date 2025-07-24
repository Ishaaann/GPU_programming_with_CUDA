# Compiler
NVCC = nvcc

# Compiler flags - using a common compute capability for broad compatibility
NVCC_FLAGS = -O3 -arch=sm_52

# Executable name
EXE = grayscale_converter

# Source files located in the src directory
SRCS = src/main.cu src/grayscale.cu

# Object files derived from source files
OBJS = $(SRCS:.cu=.o)

# Default target to build the executable
all: $(EXE)

# Link object files into the final executable
$(EXE): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compile CUDA source files into object files
src/%.o: src/%.cu src/grayscale.cuh
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Clean up build files
clean:
	rm -f $(EXE) $(OBJS)
