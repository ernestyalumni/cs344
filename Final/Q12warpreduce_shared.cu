/** 
 * Q12warpreduce_shared.cu
 * \file Q12warpreduce_shared.cu
 * \brief Udacity cs344 Quiz:Final - Question 12
 * cf. https://classroom.udacity.com/courses/cs344/lessons/2133758814/concepts/1388615750923#
 * 
 * Compilation tip
 * nvcc Q12warpreduce_shared.cu -o Q12warpreduce_shared.exe
 * or
 * nvcc -std=c++11 Q12warpreduce_shared.cu -o Q12warpreduce_shared.exe
 * to include C++11 features
 * 
 * */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "./warpreduce/part_a/compare.h"
#include "./warpreduce/part_a/gputimer.h"

// Subpart A:
// Write step 1 as a kernel that operates on threads 0--31.
// Assume that the input flags are 0 for false and 1 for true and are stored
// in a local per-thread register called p (for predicate).
//
// You have access to 31 words of shared memory s[0:31], with s[0]
// corresponding to thread 0 and s[31] corresponding to thread 31.
// You may change the values of s[0:31]. Put the return sum in s[0].
// Your code should execute no more than 5 warp-wide addition operations.

__device__ unsigned int shared_reduce(unsigned int p, volatile unsigned int * s) {
    // Assumes values in 'p' are either 1 or 0
    // Assumes s[0:31] are allocated
    // Sums p across warp, returning the result. Suggest you put
    // result in s[0] and return it
    // You may change any value in s
    // You should execute no more than 5 + operations (if you're doing
    // 31, you're doing it wrong)
    //
    // TODO: Finish the rest of this function
     
    int i_x = threadIdx.x ; // i_x = [0, 1, ... M_x-1), with M_x = 32 = ARRAY_SIZE in this case

	// load input values of d_in from global memory (through p) into shared memory
	s[i_x] = p; 

	// synchronize threads in block to ensure all thread values stored
	__syncthreads();

	int M_x = blockDim.x ; // M_x := total number of threads in a (single thread) block, ARRAY_SIZE = 32 in this case
	for (int i = (M_x >> 1); i > 0; i >>= 1) { 
		if ( i_x < i ) { 
			s[i_x] += s[i_x + i] ; }
		__syncthreads() ; 	
	}

	return s[0]; 
}

__global__ void reduce(unsigned int * d_out_shared,
                       const unsigned int * d_in)
{
    extern __shared__ unsigned int s[];
    int t = threadIdx.x;
    int p = d_in[t];
    unsigned int sr = shared_reduce(p, s);
    if (t == 0)
    {
        *d_out_shared = sr;
    }
}

int main(int argc, char **argv)
{
    const int ARRAY_SIZE = 32;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned int);

    // generate the input array on the host
    unsigned int h_in[ARRAY_SIZE];
    unsigned int sum = 0;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [0, 1]
        h_in[i] = (float)random()/(float)RAND_MAX > 0.5f ? 1 : 0;
        sum += h_in[i];
    }

    // declare GPU memory pointers
    unsigned int * d_in, * d_out_shared;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out_shared, sizeof(unsigned int));

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    GpuTimer timer;
    timer.Start();
    // launch the kernel
    reduce<<<1, ARRAY_SIZE, ARRAY_SIZE * sizeof(unsigned int)>>>
        (d_out_shared, d_in);
    timer.Stop();

    printf("Your code executed in %g ms\n", timer.Elapsed());

    unsigned int h_out_shared;
    // copy back the sum from GPU
    cudaMemcpy(&h_out_shared, d_out_shared, sizeof(unsigned int), 
               cudaMemcpyDeviceToHost);
    
    // compare your results against the sum
    compare(h_out_shared, sum);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out_shared);
}
