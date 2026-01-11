# CUDA-Kar

Welcome to **CUDA-Kar** (means someone who knows how to work with CUDA in Persian!). This repository contains my journey learning CUDA programming. I had tremondous fun learning about parralel algorithms and developing kernels considering GPU hardware constraints. So, I decided to share it with you.

Each folder contains a topic, CUDA implementation, and a detailed explanation of the algorithm in `README` file.

## Roadmap

- [x] **0. Preliminaries**: Understanding the CPU vs. GPU paradigm.
- [x] **1. Vector Addition**: Hello World of CUDA.
- [x] **2. Matrix Multiplication (GEMM)**: Naive to tiled implementation.
- [x] **3. Reduction**: Parallelizing sum/max operations.
- [x] **4. Matrix Vector Multiplication (GEMV)**: Naive to tiled implementation.
- [ ] **5. Softmax**: Numerical stability in parallel.
- [ ] **6. Attention**
- [ ] **7. Flash Attention**
- [ ] **8. Paged Attention**
- [ ] **9. Quantization**

## Prerequisites

- **C/C++ specifics**: CUDA is written in C++.
- **NVIDIA GPU**: The hardware to run the code (or a Colab instance).
- **CUDA Toolkit**: The compiler (nvcc) and drivers.

## Resources

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [NVIDIA Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)