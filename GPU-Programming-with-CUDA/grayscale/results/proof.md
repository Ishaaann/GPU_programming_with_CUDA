# Proof of Execution

This document provides evidence that the `grayscale_converter` program was successfully compiled and executed on sample data.

### **Compilation** ⚙️

The project was compiled successfully using the provided `Makefile`.

```sh
$ make
nvcc -O3 -arch=sm_52 -c -o src/main.o src/main.cu
nvcc -O3 -arch=sm_52 -c -o src/grayscale.o src/grayscale.cu
nvcc -O3 -arch=sm_52 -o grayscale_converter src/main.o src/grayscale.o
