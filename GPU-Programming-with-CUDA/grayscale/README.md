
High-Performance Batch Image Grayscaling via CUDA
==================================================

Table of Contents
-----------------
- [Abstract](#abstract)
- [System Requirements](#system-requirements)
- [Directory Structure](#directory-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Cleanup](#cleanup)

Abstract
--------
This project presents a high-performance application for batch conversion of color images to grayscale, leveraging the parallel computing architecture of NVIDIA's CUDA platform. The program processes a collection of images from a designated input directory using GPU acceleration and saves the resulting grayscale images to a specified output directory. This serves as a practical demonstration of GPU-accelerated image processing.

System Requirements
-------------------
- **NVIDIA GPU:** CUDA-capable NVIDIA graphics processing unit.
- **NVIDIA CUDA Toolkit:** Required for the `nvcc` compiler and development libraries.
- **stb Image Libraries:** `stb_image.h` and `stb_image_write.h` for image I/O.

Directory Structure
-------------------
```
/
|-- Makefile                 # Build automation
|-- src/                     # Source code and headers
|   |-- main.cu
|   |-- grayscale.cuh
|   |-- grayscale.cu
|   |-- stb_image.h          # Dependency: Download from stb repo
|   |-- stb_image_write.h    # Dependency: Download from stb repo
|-- input_data/              # Source color images
|-- output_data/             # Grayscale images
|-- results/                 # Documentation/proof of execution
|   |-- proof.md
```

Installation and Setup
----------------------

### 1. Acquire Dependencies
Download the following files from the [official stb repository](https://github.com/nothings/stb) and place them in the `src/` directory:
- `stb_image.h`
- `stb_image_write.h`

### 2. Prepare Input Data
Place all color images to be converted in the `input_data/` directory.

### 3. Compile the Project
Open a terminal in the project root and run:
```sh
make
```
This will use `nvcc` to build the project and create the `grayscale_converter` executable.

Usage
-----
To run the program and convert all images in `input_data/`:
```sh
./grayscale_converter
```
The grayscale images will be saved in the `output_data/` directory.

Cleanup
-------
To remove intermediate object files and the executable, run:
```sh
make clean
```
