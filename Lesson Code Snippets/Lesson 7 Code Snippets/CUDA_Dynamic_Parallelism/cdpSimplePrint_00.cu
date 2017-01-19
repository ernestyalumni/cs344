// cf. http://stackoverflow.com/questions/19287461/compiling-code-containing-dynamic-parallelism-fails
/** Compilation tip:
 * nvcc -arch=sm_35 -rdc=true cdpSimplePrint_00.cu -o cdpSimplePrint_00.exe
 * 
 * Explanation
 * without the -arch=sm_35 (architecture, SM 3.5) flag, I obtain this error:
 * error: calling a __global__ function("cdp_kernel") from a __global__ function("cdp_kernel") is only allowed on the compute_35 architecture or above
 * 
 * On my "GeForce GTX 980 Ti" it says 
 * CUDA Capability Major/Minor version number:    5.2
 * from running
 * cs344/Lesson Code Snippets/Lesson 5 Code Snippets/deviceQuery_simplified.exe
 * 
 * so flag -arch=sm_52 works, i.e.
 * nvcc -arch=sm_52 -rdc=true cdpSimplePrint_00.cu -o cdpSimplePrint_00.exe
 * 
 * without -rdc=true flag, I obtain error:
 * error: kernel launch from __device__ or __global__ functions requires separate compilation mode
 * rdc is --relocatable-device-code=true
 * from CUDA Toolkit Documentation:
 *  Enable (disable) the generation of relocatable device code. 
 *  If disabled, executable device code is generated. 
 *  Relocatable device code must be linked before it can be executed.
 * 
 * Allowed values for this option: true, false.
 * Default value: false
 * 
 * */

#include <iostream>
#include <cstdio>  // stderr, fprintf, printf

#include "utils.h" // checkCudaErrors


////////////////////////////////////////////////////////////////////////////////
// Variable on the GPU used to generate unique identifiers of blocks.
////////////////////////////////////////////////////////////////////////////////
__device__ int g_uids = 0;

////////////////////////////////////////////////////////////////////////////////
// Print a simple message to signal the block which is currently executing.
////////////////////////////////////////////////////////////////////////////////
__device__ void print_info(int depth, int thread, int uid, int parent_uid)
{
    if (threadIdx.x == 0)
    {
        if (depth == 0)
            printf("BLOCK %d launched by the host\n", uid);
        else
        {
            char buffer[32];

            for (int i = 0 ; i < depth ; ++i)
            {
                buffer[3*i+0] = '|';
                buffer[3*i+1] = ' ';
                buffer[3*i+2] = ' ';
            }

            buffer[3*depth] = '\0';
//            printf("%sBLOCK %d launched by thread %d of block %d\n", buffer, uid, thread, parent_uid);

			printf("%s thread %d launches BLOCK (i.e. uid) %d of block (parent_uid) %d \n", 
					buffer, thread, uid, parent_uid) ; 
 
        }
    }

    __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// The kernel using CUDA dynamic parallelism.
//
// It generates a unique identifier for each block. Prints the information
// about that block. Finally, if the 'max_depth' has not been reached, the
// block launches new blocks directly from the GPU.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_kernel(int max_depth, int depth, int thread, int parent_uid)
{
    // We create a unique ID per block. Thread 0 does that and shares the value with the other threads.
    __shared__ int s_uid;

    if (threadIdx.x == 0)
    {
        s_uid = atomicAdd(&g_uids, 1);
    }

    __syncthreads();

    // We print the ID of the block and information about its parent.
    print_info(depth, thread, s_uid, parent_uid);

    // We launch new blocks if we haven't reached the max_depth yet.
    if (++depth >= max_depth)
    {
        return;
    }

    cdp_kernel<<<gridDim.x, blockDim.x>>>(max_depth, depth, threadIdx.x, s_uid);
}

////////////////////////////////////////////////////////////////////////////////
// Variable on the GPU used to generate unique identifiers of blocks, for 
// my version of the kernels
////////////////////////////////////////////////////////////////////////////////
__device__ int g_uids_2 = 0;

////////////////////////////////////////////////////////////////////////////////
// Print a simple message to signal the block which is currently executing;
// this is my version
////////////////////////////////////////////////////////////////////////////////
__device__ void print_info_2(int depth, int thread, int uid, int parent_uid)
{
    if (threadIdx.x == 0)
    {
        if (depth == 0)
            printf("BLOCK %d launched by the host\n", uid);
        else
        {
            char buffer[32];

            for (int i = 0 ; i < depth ; ++i)
            {
                buffer[3*i+0] = '|';
                buffer[3*i+1] = ' ';
                buffer[3*i+2] = ' ';
            }

            buffer[3*depth] = '\0';  // '\0' is termination of the string or char array

//			printf("%s thread %d launches BLOCK (i.e. uid) %d of block (parent_uid) %d \n", 
//					buffer, thread, uid, parent_uid) ; 
			printf("%s depth %d thread %d launches BLOCK (i.e. uid) %d of block (parent_uid) %d \n", 
					buffer, depth, thread, uid, parent_uid) ; 
			
        }
    }

    __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// My version of cdp_kernel, cdp_kernel_2
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_kernel_2(int max_depth, int depth, int thread, int parent_uid)
{
    // We create a unique ID per block. Thread 0 does that and shares the value with the other threads.
	__shared__ int s_uid_2;
	
	if (threadIdx.x == 0) {
		s_uid_2 = atomicAdd(&g_uids_2, 1);
	}
	__syncthreads();
	
	// We print the ID of the block and information about its parent.
	print_info_2(depth, thread, s_uid_2, parent_uid);
	
	++depth;
	if (depth >= max_depth) { return; }
	
	cdp_kernel_2<<<gridDim.x,blockDim.x>>>(max_depth,depth, threadIdx.x, s_uid_2);
}


////////////////////////////////////////////////////////////////////////////////
// Variable on the GPU used to generate unique identifiers of blocks, for 
// my version of the kernels, with blockIdx.x print out 
////////////////////////////////////////////////////////////////////////////////
__device__ int g_uids_3 = 0;

////////////////////////////////////////////////////////////////////////////////
// Print a simple message to signal the block which is currently executing;
// this is my version, with blcokIdx.x print out
////////////////////////////////////////////////////////////////////////////////
__device__ void print_info_3(int depth, int thread, int uid, int parent_uid, int blockIndex)
{
    if (threadIdx.x == 0)
    {
        if (depth == 0)
            printf("BLOCK %d launched by the host\n", uid);
        else
        {
            char buffer[32];

            for (int i = 0 ; i < depth ; ++i)
            {
                buffer[3*i+0] = '|';
                buffer[3*i+1] = ' ';
                buffer[3*i+2] = ' ';
            }

            buffer[3*depth] = '\0';  // '\0' is termination of the string or char array

//			printf("%s thread %d launches BLOCK (i.e. uid) %d of block (parent_uid) %d \n", 
//					buffer, thread, uid, parent_uid) ; 
			printf("%s thread %d launches BLOCK (i.e. uid) %d of block (parent_uid) %d for threadIdx.x = %d , blockIdx.x = %d , called from blockIndex = %d \n", 
					buffer, thread, uid, parent_uid, threadIdx.x, blockIdx.x, blockIndex) ; 
			
        }
    }

    __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// My version of cdp_kernel, cdp_kernel_2
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_kernel_3(int max_depth, int depth, int thread, int parent_uid, int blockIndex)
{
    // We create a unique ID per block. Thread 0 does that and shares the value with the other threads.
	__shared__ int s_uid;
	
	if (threadIdx.x == 0) {
		s_uid = atomicAdd(&g_uids_3, 1);
	}
	__syncthreads();
	
	// We print the ID of the block and information about its parent.
	print_info_3(depth, thread, s_uid, parent_uid, blockIndex);
	
	++depth;
	if (depth >= max_depth) { return; }
	
	cdp_kernel_3<<<gridDim.x,blockDim.x>>>(max_depth,depth, threadIdx.x, s_uid, blockIdx.x);
}



////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("starting Simple Print (CUDA Dynamic Parallelism)\n");

    // Parse a few command-line arguments.
    int max_depth = 2;  // originally 2
    
       // Print a message describing what the sample does.
    printf("***************************************************************************\n");
    printf("The CPU launches 2 blocks of 2 threads each. On the device each thread will\n");
    printf("launch 2 blocks of 2 threads each. The GPU we will do that recursively\n");
    printf("until it reaches max_depth=%d\n\n", max_depth);
    printf("In total 2");
    int num_blocks = 2, sum = 2;

    for (int i = 1 ; i < max_depth ; ++i)
    {
        num_blocks *= 4;
        printf("+%d", num_blocks);
        sum += num_blocks;
    }

    printf("=%d blocks are launched!!! (%d from the GPU)\n", sum, sum-2);
    printf("***************************************************************************\n\n");

    // We set the recursion limit for CDP to max_depth.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);

    // Launch the kernel from the CPU.
    printf("Launching cdp_kernel() with CUDA Dynamic Parallelism:\n\n");
    cdp_kernel<<<2, 2>>>(max_depth, 0, 0, -1);
    checkCudaErrors(cudaGetLastError());

    // Finalize.
    checkCudaErrors(cudaDeviceSynchronize());

	///////////////////////////////////////////////////////////////////
	// MY VERSION/modifications that I made to teach myself about 
	// DYNAMIC PARALLELISM
	//////////////////////////////////////////////////////////////////

	printf("experimenting (playing) with Simple Print (CUDA Dynamic Parallelism)\n");

	std::cout << "Input in the new max_depth (maximum depth)" << std::endl;
	int max_depth_2 = 2;
	std::cin >> max_depth_2; 
	
	// We set the recursion limit for CDP to max_depth.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth_2);

    // Launch the kernel from the CPU.
    printf("Launching cdp_kernel_2() with CUDA Dynamic Parallelism:\n\n");
    cdp_kernel_2<<<2, 2>>>(max_depth_2, 0, 0, -1);
    checkCudaErrors(cudaGetLastError());

    // Finalize.
    checkCudaErrors(cudaDeviceSynchronize());
	
	///////////////////////////////////////////////////////////////////
	// MY VERSION/modifications that I made to teach myself about 
	// DYNAMIC PARALLELISM; I also print out blockIdx
	//////////////////////////////////////////////////////////////////

	printf("experimenting (playing) with Simple Print (CUDA Dynamic Parallelism); print out blockIdx as well \n");

	std::cout << "Input in the new max_depth (maximum depth)" << std::endl;
	int max_depth_3 = 2;
	std::cin >> max_depth_3; 
	
	// We set the recursion limit for CDP to max_depth.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth_3);

    // Launch the kernel from the CPU.
    printf("Launching cdp_kernel_3() with CUDA Dynamic Parallelism:\n\n");
    cdp_kernel_3<<<2, 2>>>(max_depth_3, 0, 0, -1,0);
    checkCudaErrors(cudaGetLastError());

    // Finalize.
    checkCudaErrors(cudaDeviceSynchronize());




    exit(EXIT_SUCCESS);
}
