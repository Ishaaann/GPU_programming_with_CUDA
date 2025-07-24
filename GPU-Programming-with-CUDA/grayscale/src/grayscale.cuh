#ifndef GRAYSCALE_CUH_
#define GRAYSCALE_CUH_

#include <cuda_runtime.h>

// Declaration of the wrapper function that launches the CUDA kernel.
// This function is callable from standard C++ code (the host).
void convertToGrayscale(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);

#endif // GRAYSCALE_CUH_
