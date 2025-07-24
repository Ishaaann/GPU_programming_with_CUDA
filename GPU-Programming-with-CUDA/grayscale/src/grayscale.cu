#include "grayscale.cuh"

// CUDA Kernel: Converts an RGB image to grayscale in parallel.
// Each thread processes a single pixel.
__global__ void grayscale_kernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    // Calculate the global thread ID for the pixel
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within the image boundaries
    if (col < width && row < height) {
        // Calculate the linear index for the pixel in the 1D array
        int pixel_idx = row * width + col;
        int rgb_idx = pixel_idx * channels;

        // Get the RGB values for the current pixel
        unsigned char r = input[rgb_idx];
        unsigned char g = input[rgb_idx + 1];
        unsigned char b = input[rgb_idx + 2];

        // Apply the luminosity formula for grayscale conversion
        // Gray = 0.299*R + 0.587*G + 0.114*B
        unsigned char gray_value = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

        // Write the result to the output array
        output[pixel_idx] = gray_value;
    }
}

// Wrapper function to configure and launch the kernel
void convertToGrayscale(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    // Define the grid and block dimensions for the kernel launch
    // We use 2D blocks of 16x16 threads.
    dim3 threadsPerBlock(16, 16);
    
    // Calculate the number of blocks needed in each dimension
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    grayscale_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, channels);
}
