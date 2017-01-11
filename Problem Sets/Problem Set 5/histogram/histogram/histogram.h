/** 
 * histogram.h
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
#ifndef __HISTOGRAM_H__
#define __HISTOGRAM_H__

#include "../common/checkerror.h"

// GPU with atomics; i.e.
// GPU with (global) atomics
/****************************************/
/* GPU HISTOGRAM, GLOBAL VERSION  */
/****************************************/

__global__ void histogram_global( const unsigned int const *d_vals, unsigned int *d_Histo, 
	const unsigned int numBins, const unsigned int numElems ) ;

// GPU with atomics in shared memory with final summation of partial histograms
/****************************************/
/* GPU HISTOGRAM SHARED MEMORY  */
/****************************************/ 
__global__ void histogram_shared( const unsigned int const *d_vals, unsigned int *d_Histo, 
	const unsigned int numBins, const unsigned int numElems ) ;


// shared atomics, fast histogram
// cf. https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
// 1st phase
__global__ void histogram_smem_atomics( const unsigned int* const d_vals, //INPUT
					unsigned int* PartialHisto ,
//                  unsigned int* const d_histo,      //OUTPUT
                  const unsigned int numBins,
                  const unsigned int numElems ) ;

// shared atomics, fast histogram 
// cf. https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
// 2nd phase
__global__ void histogram_final_accum( 
							const unsigned int *PartialHisto, 
							unsigned int *d_Histo,
							unsigned int L_x, // total number of thread blocks
							const unsigned int numBins) ;

void histogram_shared_atomics_kernel( const unsigned int const *d_vals, unsigned int *d_Histo, 
	const unsigned int numBins, const unsigned int numElems,
	const unsigned int blockSize, const unsigned int blockSize2) ;



#endif // __HISTOGRAM_H__

