## Reduction

## Problem Definition

Given an array of $N$ elements, any operation transforming data elements into a single, summary value, is called a reduction operation. For example, `sum`, `min`, or `max` are all reduction operations. These operations are common and important data parallel primitives and deep learning. We will focus on `sum` reduction in this section but the same principles apply to other operations.

## Parallel Thinking

By nature, the problem is not organically parallelizable like matrix multiplication or vector addition. `sum` is usually think of as a sequential problem. However, **the final sum can be sum of partial sums computed in parallel**. Those partial sums again can be computed in parallel by sub-partitions of the array and so on. There is a tree-based structure to the problem that can be exploited to parallelize it.

![Reduction Tree](../images/reduction_tree.png)

As shown in the figure, the computation carried under sub-trees of each node are independent of each other. This means that we can compute the partial sums in parallel. 

## CUDA Implementation

Now let's think about how we can implement this in CUDA and what calculation each thread should perform. The main challenge is there is not much to do for each thread. It's more about how to organize the threads to commute with each other to compute the partial sums.

We learned that threads in the same block can be synchronized using communicate with each other using shared memory and `__syncthreads()`. Each thread can writes its partial sum to shared memory and then all threads can read from shared memory to compute the final sum. But what if the array is too large to fit in shared memory? In that case thread blocks should be synchronized and communicate partial sums to each other. **But CUDA has no global synchronization.**

![Recursive Launch](../images/recursive_launch.png)

The solution is to decompose the full sum into multiple kernels and launch them recursively. In the case of reductions, code for all levels is the same.

## Kernel Code

![Interleaved Addressing](../images/interleaved_addressing.png)

So whithin each block, we use shared memory to store the partial sums and then use `__syncthreads()` to synchronize the threads. At step 1, each thread $i$ adds its element to the element stored in shared memory by the thread at the next index $i+1$ (stride 1). At step 2, threads with even indices $i$ adds its element to the element stored in shared memory by the thread at the next index $i+2$ (stride 2). This process continues till the thread with index 0 has computed the final sum, as shown in the figure.

```cpp
__global__ void reduce(float *input, float *output) {

    // shared memory for partial sums
    extern __shared__ float sdata[];

    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();

    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```