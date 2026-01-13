#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__,          \
              __LINE__, err, cudaGetErrorString(err));                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void reduce(float *g_idata, float *g_odata, unsigned int n) {
  // shared memory for partial sums
  extern __shared__ float sdata[];

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  // Load two elements per thread and perform first add of the reduction
  // We add boundary checks to handle array sizes that are not multiples of the
  // block size
  float mySum = (i < n) ? g_idata[i] : 0.0f;
  if (i + blockDim.x < n) {
    mySum += g_idata[i + blockDim.x];
  }

  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  // Optimization: Sequential addressing
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Optimization: Unroll the last warp
  if (tid < 32) {
    volatile float *vsmem = sdata;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

int main() {
  int N = 1 << 24; // 16M elements
  printf("Reducing array of size %d\n", N);
  size_t bytes = N * sizeof(float);

  // Allocate host memory
  float *h_in = (float *)malloc(bytes);

  // Initialize
  for (int i = 0; i < N; i++)
    h_in[i] = 1.0f;

  // Allocate device memory
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes)); // Allocate output buffer (safe size)

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  // Kernel configuration
  int threads = 256;
  // Each thread process 2 elements, so each block process threads*2 elements
  int blocks = (N + (threads * 2 - 1)) / (threads * 2);

  // Create events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Multi-pass reduction loop
  int curr_n = N;
  int curr_blocks = blocks;

  // Pointers to swap
  float *d_current_in = d_in;
  float *d_current_out = d_out;

  while (curr_n > 1) {
    // Launch kernel
    reduce<<<curr_blocks, threads, threads * sizeof(float)>>>(
        d_current_in, d_current_out, curr_n);
    CUDA_CHECK(cudaGetLastError());

    // Update n and blocks for next pass
    curr_n = curr_blocks;
    curr_blocks = (curr_n + (threads * 2 - 1)) / (threads * 2);

    // Swap pointers
    float *temp = d_current_in;
    d_current_in = d_current_out;
    d_current_out = temp;
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // After the loop, the result is in *d_current_in (the input of the
  // theoretical next step)
  float h_result;
  CUDA_CHECK(cudaMemcpy(&h_result, d_current_in, sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("Sum: %f\n", h_result);
  printf("Time: %f ms\n", milliseconds);

  // Verification
  float expected = (float)N;
  if (std::abs(h_result - expected) < 1e-5) {
    printf("Verification PASSED\n");
  } else {
    printf("Verification FAILED (Expected: %f, Got: %f)\n", expected, h_result);
  }

  // Cleanup
  free(h_in);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
