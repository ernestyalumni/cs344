/** 
 * main.cu
 * 
 * \file main.cu
 * \author Ernest Yeung
 * \brief demonstrating histogram methods, from serial to shared atomics
 * 
 * typed up by Ernest Yeung  ernestyalumni@gmail.com
 * \date 20170110
 * cf. https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
 * GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell
 * Nikolay Sakharnykh
 * 
 * Also look here for the experimental version
 * cf. https://github.com/NVlabs/cub/blob/master/experimental/histogram/histogram_smem_atomics.h
 * cub/experimental/histogram/histogram_smem_atomics.h
 * 
 * Also in 
 * cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming 
 * Chapter 9 Atomics 
 * 9.4 Computing Histograms
 * 9.4.1 CPU Histogram Computation
 * and 
 * cf. https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/histshared.cu
 * 
 * 
 *  
 * Compilation tip
 * nvcc -std=c++11 main.cu -o main.o -dc
 *  
 * 
 */
#include "histogram.h"

// GPU with atomics; i.e.
// GPU with (global) atomics
/****************************************/
/* GPU HISTOGRAM GLOBAL   */
/****************************************/
__global__ void histogram_global( const unsigned int const *d_vals, unsigned int *d_Histo, 
	const unsigned int numBins, const unsigned int numElems ) {

	int k_x = threadIdx.x + blockDim.x * blockIdx.x ; // k_x = 0,1,... numElems-1
	int stride = blockDim.x * gridDim.x ; 

	for (unsigned int i = k_x; i < numElems; i += stride ) { 
		atomicAdd(&d_Histo[ d_vals[i]], 1) ; 
	}
}


// GPU with atomics in shared memory with final summation of partial histograms
/****************************************/
/* GPU HISTOGRAM SHARED MEMORY  */
/****************************************/ 

__global__ void histogram_shared( const unsigned int const *d_vals, unsigned int *d_Histo, 
	const unsigned int numBins, const unsigned int numElems ) {
		
	int k_x = threadIdx.x + blockDim.x * blockIdx.x ; // k_x = 0, 1, ... numElems-1 
	int i_x = threadIdx.x ; 
	int M_x = blockDim.x  ; 
	int offset = blockDim.x * gridDim.x ; 
	
	extern __shared__ unsigned int s[] ; // |s| = numBins ; i.e. size of s, shared memory, is numBins
	
	for (unsigned int i = i_x; i < numBins; i += M_x ) {
		s[i] = 0; }
	__syncthreads(); 	
	
	for (unsigned int i = k_x; i < numElems; i += offset) {
		atomicAdd( &s[ d_vals[i] ], 1) ;
	}
	
	__syncthreads(); // ensure last of our writes have been committed

	for (unsigned int i = i_x; i < numBins; i += M_x ) {
		atomicAdd( &(d_Histo[ i ]), s[ i ] ) ;
	}
	__syncthreads(); 	

		
}

// shared atomics, fast histogram 
// cf. https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
// 1st phase
__global__ void histogram_smem_atomics( const unsigned int* const d_vals, //INPUT
					unsigned int* PartialHisto ,
//                  unsigned int* const d_histo,      //OUTPUT
                  const unsigned int numBins,
                  const unsigned int numElems ) {
	
	// global position
	int k_x = threadIdx.x + blockDim.x * blockIdx.x ; // k_x = [0, numElems)
	
	// threads (thread position) in workgroup
	int i_x = threadIdx.x ;				  
	
	// group index in 0 .. ngroups -1, i.e. 
	// thread block index j_x in 0 .. L_x-1 blocks on a grid in x-direction, where L_x = (numElems + M_x - 1)/M_x
	int j_x = blockIdx.x ; 
	int M_x = blockDim.x ; 
	int offset = blockDim.x * gridDim.x ; 
	
	// initialize smem
	//__shared__ unsigned int smem[ numBins ] ;
	extern __shared__ unsigned int smem[] ;  // |smem| = numBins ; i.e. size of smem, shared memory, is numBinss
		
	for (int i = i_x; i < numBins ; i += M_x) { 
		smem[i] = 0 ; }
	__syncthreads(); 
	
	// process input values
	// updates our group (i.e. (single) thread block)'s partial histogram in smem
	for (int pos = k_x; pos < numElems; pos += offset ) { 
//		unsigned int temp_val = d_vals[pos] ; 
//		atomicAdd( &smem[ temp_val ], 1);
		atomicAdd( &smem[ d_vals[pos] ], 1 ) ;
	}			  
	__syncthreads();
	
	for (int i = i_x; i < numBins; i += M_x ) {
		PartialHisto[i + numBins * j_x] = smem[ i  ] ; 
	}	
					  
}

// shared atomics, fast histogram 
// cf. https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
// 2nd phase
__global__ void histogram_final_accum( 
							const unsigned int *PartialHisto, 
							unsigned int *d_Histo,
							unsigned int L_x, // total number of thread blocks
							const unsigned int numBins) {
	int k_x = threadIdx.x + blockIdx.x * blockDim.x ;  // k_x = 0, 1, ... numBins-1
	if (k_x  >= numBins) { return; }

//	unsigned int total = 0;
	int total = 0 ;
	for (int j = 0; j < L_x ; ++j) {
		total += PartialHisto[ k_x + j * numBins ] ; 
	}
	d_Histo[k_x] = total;
}

void histogram_shared_atomics_kernel( const unsigned int const *d_vals, unsigned int *d_Histo, 
	const unsigned int numBins, const unsigned int numElems, 
	const unsigned int blockSize, const unsigned int blockSize2) { 
	
	unsigned int gridSize = (numElems + blockSize - 1)/blockSize ; 
	
	unsigned int * d_PartialHisto ; // Partial Histograms, of size numBins * gridSize 
	
	checkCudaErrors( cudaMalloc( &d_PartialHisto, sizeof(unsigned int) * numBins * gridSize ) ) ;
	checkCudaErrors( cudaMemset( d_PartialHisto, 0, sizeof(unsigned int) * numBins * gridSize));
	
	histogram_smem_atomics<<<gridSize,blockSize, numBins * sizeof(unsigned int)>>>(d_vals, d_PartialHisto, numBins, numElems) ;
	
	histogram_final_accum<<<numBins, blockSize2>>>(d_PartialHisto, d_Histo, gridSize, numBins) ;
	
		
	checkCudaErrors( cudaFree( d_PartialHisto) ); 

}



