#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h> // Required for directory creation

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "grayscale.cuh"

// Helper function to check for CUDA errors
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
        exit(1);
    }
}

// Function to create a directory if it doesn't exist
void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0755);
    } else if (!(info.st_mode & S_IFDIR)) {
        fprintf(stderr, "Error: %s is not a directory!\n", path.c_str());
        exit(1);
    }
}


int main() {
    const std::string input_dir = "input_data";
    const std::string output_dir = "output_data";

    // Create output directory if it doesn't exist
    createDirectory(output_dir);

    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(input_dir.c_str())) != NULL) {
        std::cout << "Starting batch image processing..." << std::endl;

        // Iterate over all the files in the input directory
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename == "." || filename == "..") {
                continue;
            }

            std::string input_path = input_dir + "/" + filename;

            // 1. Load image from disk using stb_image
            int width, height, channels;
            unsigned char *h_input_img = stbi_load(input_path.c_str(), &width, &height, &channels, 0);

            if (h_input_img == NULL) {
                std::cerr << "Error loading image: " << input_path << std::endl;
                continue;
            }

            // We only process RGB or RGBA images
            if (channels < 3) {
                std::cout << "Skipping non-color image: " << filename << std::endl;
                stbi_image_free(h_input_img);
                continue;
            }

            std::cout << "Processing: " << filename << " (" << width << "x" << height << ")" << std::endl;

            // 2. Allocate memory on the host and device
            size_t img_size = width * height * channels * sizeof(unsigned char);
            size_t gray_img_size = width * height * sizeof(unsigned char);

            unsigned char *h_output_gray_img = (unsigned char*)malloc(gray_img_size);
            unsigned char *d_input_img, *d_output_gray_img;

            checkCuda(cudaMalloc((void**)&d_input_img, img_size));
            checkCuda(cudaMalloc((void**)&d_output_gray_img, gray_img_size));

            // 3. Copy image data from host to device
            checkCuda(cudaMemcpy(d_input_img, h_input_img, img_size, cudaMemcpyHostToDevice));

            // 4. Launch the CUDA kernel
            convertToGrayscale(d_input_img, d_output_gray_img, width, height, channels);

            // Check for any errors during kernel execution
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            // 5. Copy the result back from device to host
            checkCuda(cudaMemcpy(h_output_gray_img, d_output_gray_img, gray_img_size, cudaMemcpyDeviceToHost));

            // 6. Save the grayscale image to disk
            std::string output_path = output_dir + "/gray_" + filename;
            // Save as PNG to avoid compression artifacts and handle single channel easily
            size_t dot_pos = output_path.find_last_of(".");
            if (dot_pos != std::string::npos) {
                 output_path.replace(dot_pos, output_path.length() - dot_pos, ".png");
            } else {
                 output_path += ".png";
            }
            stbi_write_png(output_path.c_str(), width, height, 1, h_output_gray_img, width * sizeof(unsigned char));

            // 7. Free memory
            stbi_image_free(h_input_img);
            free(h_output_gray_img);
            cudaFree(d_input_img);
            cudaFree(d_output_gray_img);
        }
        closedir(dir);
        std::cout << "Processing complete. Results are in the 'output_data' directory." << std::endl;
    } else {
        std::cerr << "Error: Could not open input directory '" << input_dir << "'. Please create it and add images." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
