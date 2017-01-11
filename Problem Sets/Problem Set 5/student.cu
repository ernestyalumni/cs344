/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h> // thrust::device_ptr


/** 
 * global version of histogram 
 * 
 * on GeForce GTX 980 Ti, 4.338208 msecs 4.589024 msecs 4.319296 msec. 
**/
/**
__global__
void yourHisto(const unsigned int* const vals, //INPUT
           unsigned int* const d_histo,      //OUPUT
           const unsigned int numVals)
{
    unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
//	if (id >= numVals) { return ; } // sanity check for accessing the input values in global memory
	int stride = blockDim.x * gridDim.x;
	
	for (unsigned int i = id; i < numVals; i += stride) { 
		unsigned int bin = vals[i];
		atomicAdd(&(d_histo[bin]),1);
	}
}
*/

/** 
 * shared version of histogram 
 * 
 * on GeForce GTX 980 Ti, 0.569824 msecs 0.570016 msecs 0.570144 msec. 
**/
// GPU with atomics in shared memory with final summation of partial histograms
/****************************************/
/* GPU HISTOGRAM SHARED MEMORY  */
/****************************************/ 

__global__ void yourHisto( const unsigned int* const d_vals, unsigned int *d_Histo, 
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

////////////////////////////////////////////////////////////////////////
/** *******************************************************************/
/** GPU shared atomics fast histogram
 *  
 * on GeForce GTX 980 Ti, 1.905600 msecs 1.879360 msecs 1.914016 msec. 
 * 
 * ********************************************************************/
////////////////////////////////////////////////////////////////////////

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

void histogram_shared_atomics_kernel( const unsigned int* const d_vals, unsigned int *d_Histo, 
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

/** *******************************************************************/
/** END OF GPU shared atomics fast histogram
 * ********************************************************************/
////////////////////////////////////////////////////////////////////////


		



void computeHistogram(const unsigned int* const d_vals, //INPUT
                  unsigned int* const d_histo,      //OUTPUT
                  const unsigned int numBins,
                  const unsigned int numElems)
{
	/**
	 * *********** block, grid dimensions *************
	 * */	

	// histogram global
//    const unsigned int BLOCK_SIZE = 256; 
//	const unsigned int GRID_SIZE  = (numElems + BLOCK_SIZE - 1)/ BLOCK_SIZE ; 

	// histogram shared memory
	const unsigned int blocksize_s = 1024;  // numBins  = 1024
	const unsigned int gridsize_s  = (numElems + blocksize_s - 1)/blocksize_s ; 


	// histogram, shared atomics
// 1st. phase - input values from global memory into shared memory local histograms
	// notice that gridsize1 will be necessary for size of PartialHisto
	const unsigned int blocksize1 = 1024;  // it is correct for blocksize1 = 256, block_accum = 64; or 512, 64, resp.; or 1024, 1024, resp.
	
	// 2nd. phase - merge, concatenate, accumulate - block, grid dimensions
	const unsigned int block_accum = 1024 ; // it is correct for block_accum = 64, blocksize1 = 256, or 64, 512, resp.; or 1024, 1024, resp.

	/** END of block, grid dimensions *****************/
	// END of block, grid dimensions


/** 
 * global version of histogram 
 * on GeForce GTX 980 Ti, 4.338208 msecs 4.589024 msecs 4.319296 msec. 
 * 
    yourHisto<<<GRID_SIZE, BLOCK_SIZE>>>(d_vals,d_histo,numElems);
*/


/** 
 * shared memory version of histogram 
 * on GeForce GTX 980 Ti, 0.569824 msecs 0.570016 msecs 0.570144 msec. for blocksize_s = 1024
 * */
	yourHisto<<<gridsize_s, blocksize_s, numBins * sizeof(unsigned int)>>>(d_vals,d_histo,numBins,numElems);


/** 
 * on GeForce GTX 980 Ti, 1.905600 msecs 1.879360 msecs 1.914016 msec. 
 *
 * ********************************************************************
 * shared atomics version of histogram 
 * ********************************************************************/
/*
	histogram_shared_atomics_kernel( d_vals, d_histo, numBins, numElems, 
		blocksize1, block_accum) ; 
*/


}
