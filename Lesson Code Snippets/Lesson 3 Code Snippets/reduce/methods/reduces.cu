/* reduces.cu
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates reduce with parallel and serial implementations
 * with CUDA C/C++ and global memory
 * 
 * */
#include "reduces.h"

// parallel implementations

__global__ void global_reduce_kernel( float * d_out, float * d_in) 
{
	int k_x = threadIdx.x + blockDim.x * blockIdx.x ; 
	int tid = threadIdx.x ;
	
	// do reduction in global mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			d_in[k_x] += d_in[k_x + s];
		}
		__syncthreads(); 		// make sure all adds at one stage are done!
	}
	
	// only thread 0 writes result for this block back to global mem
	if (tid == 0) 
	{
		d_out[blockIdx.x] = d_in[k_x];
	}
}

__global__ void shmem_reduce_kernel(float * d_out, const float * d_in)
{
	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ float sdata[];
	
	int k_x = threadIdx.x + blockDim.x + blockIdx.x ;
	int tid = threadIdx.x ;
	
	// load shared mem from global mem
	sdata[tid] = d_in[k_x] ;
	__syncthreads();			// make sure entire block is loaded!
	
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads() ;			// make sure all adds at one stage are done!
	}
	
	// only thread 0 writes result for this block back to global mem
	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}

void reduce_global(float * d_in, float * out, const int L, int M_in) 
{
	int N_x { ( L + M_in - 1)/ M_in } ; 
	int M_x { M_in };
	
	
	// declare GPU memory pointers
	float *dev_intermediate, *dev_out;
	
	// allocate GPU memory
	checkCudaErrors( 
		cudaMalloc((void **) &dev_out, sizeof(float)) );
	checkCudaErrors( 
		cudaMalloc((void **) &dev_intermediate, N_x * sizeof(float)) );
	
	global_reduce_kernel<<<N_x, M_x>>>( dev_intermediate, d_in ) ;
	
	// now we're down to one block left, so reduce it
	M_x = N_x;
	N_x = 1;

	global_reduce_kernel<<<N_x,M_x>>>( dev_out, dev_intermediate) ;

		// copy our results from device to host 
	checkCudaErrors(
		cudaMemcpy( out, dev_out, sizeof(float), cudaMemcpyDeviceToHost) );

	cudaFree( dev_out );
	cudaFree( dev_intermediate );
}

void reduce_shmem(float * d_in, float * out, const int L, int M_in) 
{
	int N_x { ( L + M_in - 1)/ M_in } ; 
	int M_x { M_in };
	
	// declare GPU memory pointers
	float *dev_intermediate, *dev_out;
	
	// allocate GPU memory
	checkCudaErrors( 
		cudaMalloc((void **) &dev_out, sizeof(float)) );
	checkCudaErrors( 
		cudaMalloc((void **) &dev_intermediate, N_x * sizeof(float)) );
	
	shmem_reduce_kernel<<<N_x, M_x, M_x*sizeof(float)>>>( dev_intermediate, d_in ) ;
	
	// now we're down to one block left, so reduce it
	M_x = N_x;
	N_x = 1;

	shmem_reduce_kernel<<<N_x,M_x, M_x*sizeof(float)>>>( dev_out, dev_intermediate) ;

		// copy our results from device to host 
	checkCudaErrors(
		cudaMemcpy( out, dev_out, sizeof(float), cudaMemcpyDeviceToHost) );

	cudaFree( dev_out );
	cudaFree( dev_intermediate );
}
