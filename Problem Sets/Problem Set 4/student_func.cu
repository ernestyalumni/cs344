//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

/* d_Histo \in (\mathbb{Z}^+)^2, indeed, d_Histo only has 2 elements, 
 * d_Histo[0] = how many 0 bits of all the d_in
 * d_Histo[1] = how many 1 bits of all the d_in
 * */

#include <thrust/device_ptr.h>



__global__ void histo_kernel( unsigned int * const d_in, unsigned int * d_Histo, 
	const int numElems, unsigned int bit) {
	const int numBits = 1;
	const int numBins = 1 << numBits; 
	unsigned int mask = (numBins - 1) << bit; 
		
	int k_x = threadIdx.x + blockDim.x * blockIdx.x ; 
	if (k_x >= numElems) { return; }
	unsigned int bin = (d_in[k_x] & mask ) >> bit;		
	atomicAdd(&d_Histo[bin], 1);		
}


// single block (exclusive) scan 
__global__ void single_scan_kernel(unsigned int * d_in, const size_t numBins, 
	const int numElems) {
	
	int i_x = threadIdx.x ;
	if (i_x >= numElems) { return ; }
	extern __shared__ float sh_tmp[];
	sh_tmp[i_x] = d_in[i_x];
	__syncthreads();
	
	for (int d=1; d< numBins; d *= 2 ) {
		if (i_x>= d) {
			sh_tmp[i_x] += sh_tmp[i_x - d];
		}
		__syncthreads();
	}
	if (i_x==0) { d_in[0] = 0; }
	else { d_in[i_x] = sh_tmp[i_x -1]; } ; // inclusive->exclusive
}


// d_scan size,i.e. |d_scan|= numElems = |d_in|
__global__ void flipscan_kernel(unsigned int * d_in, unsigned int *d_scan,
	const int numElems, unsigned int bit) {
	const int numBits = 1;
	const int numBins = 1 << numBits; 
	unsigned int mask = (numBins - 1) << bit; 
	int k_x = threadIdx.x + blockDim.x * blockIdx.x ;
	if (k_x >= numElems) { return ; }
	unsigned int bin = (d_in[k_x] & mask) >> bit;
	d_scan[k_x] = bin ? 0 : 1;
	
}


// d_scan is of length numElems,i.e. |d_scan|=numElems
__global__ void scatter_kernel(unsigned int * const d_inputVals, 
								unsigned int* const d_inputPos,
								unsigned int* const d_outputVals,
								unsigned int* const d_outputPos,
								const size_t numElems,
								unsigned int* const d_Histo,
								unsigned int* const d_scan,
								unsigned int bit) {
	const int numBits = 1;
	const int numBins = 1 << numBits; 
	unsigned int mask = (numBins - 1) << bit; 
	int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	if (k_x >= numElems) { return ; }
	
	int target_id = 0;
	unsigned int bin = (d_inputVals[k_x] & mask) >> bit;
	if ( bin) {
		target_id = k_x + d_Histo[1] - d_scan[k_x];  // k_x + (number of 1-bits.) - (position obtained from scanning the predicates of 0,1)
	} 
	else {
		target_id = d_scan[k_x];
	}
	d_outputVals[target_id] = d_inputVals[k_x];
	d_outputPos[target_id]  = d_inputPos[k_x];
										
}
				
/** ********************************************************************
 ** **************** EXCLUSIVE SCAN, BLELLOCH SCAN *********************
 * ********************************************************************/
template <typename T>
__global__ void copy_swap(T * f_in, T * f_target, const int N) { 
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x ; 
	
	if (k_x >= N) { return ; }
	
	T tempval = f_in[k_x];
	f_in[k_x] = f_target[k_x] ; 
	f_target[k_x] = tempval;
}


template<typename T>
__global__ void Blelloch_up_global( T* f_in, T* f_out, const int k, const int L_x) {

	const int k_x = threadIdx.x + blockIdx.x * blockDim.x ;
	const int offset = 1 << k; // offset = 2^k
	
	// check if global thread index happens to fall out of global length of desired, target, input array
	if (k_x >= L_x) {
		return ; }

	T tempval;
	// k_x = 2^kj-1, j \in \lbrace 1,2,\dots \lfloor N/2^k \rfloor \rbrace check
	if ( ((k_x%offset)==(offset-1)) && (k_x >= (offset - 1)) ) { 
		tempval = f_in[k_x] + f_in[k_x-offset/2]; }
	else {
		tempval = f_in[k_x] ; }
	f_out[k_x] = tempval;
}

template<typename T>
__global__ void Blelloch_down_global( T* f_in, T* f_out, const int k, const int L_x) {

	const int k_x = threadIdx.x + blockIdx.x * blockDim.x ;
	const int offset = 1 << k ; // offset = 2^k
	
	// check if global thread index happens to fall out of global length of desired, target, input array
	if (k_x >= L_x) {
		return ;}
		
	// k_x = 2^kj-1, j \in \lbrace \lfloor N/2^k \rfloor,... 2,1 \rbrace check
	if ( ((k_x%offset)==(offset-1)) && (k_x >= (offset-1)) ) {
		T tempval_switch = f_in[k_x] ;
		T tempval = f_in[k_x] + f_in[k_x-offset/2] ;
		f_out[k_x] = tempval ;
		f_out[k_x-offset/2] = tempval_switch; }
	else {
		T tempval = f_in[k_x]; 
		f_out[k_x] = tempval ; }
}

template<typename T, T identity>
//void Blelloch_scan_kernelLauncher(T* dev_f_in, const unsigned int L_x, 
//									const unsigned int M_in) {
void Blelloch_scan_kernelLauncher(T* dev_f_in, T* dev_f_out, 
				const unsigned int L_x, const unsigned int M_in) {

//	auto Nb = static_cast<int>(std::log2( L_x) );
	unsigned int Nb  = ((unsigned int) log2( ((double) L_x)) );

/*	
	T* dev_f_out;
	checkCudaErrors( 
		cudaMalloc( &dev_f_out, L_x*sizeof(T) ));
	checkCudaErrors(
		cudaMemset( dev_f_out, 0, L_x*sizeof(T) ));
*/

	// determine number of thread blocks to be launched
	const unsigned int N_x  =  ( L_x + M_in - 1 ) / M_in  ;

	// do up sweep
	for (unsigned int k = 1; k <= Nb; ++k) {
		Blelloch_up_global<T><<<N_x, M_in>>>( dev_f_in, dev_f_out, k, L_x) ; 
		copy_swap<T><<<N_x,M_in>>>(dev_f_in, dev_f_out,L_x); }
		
	// crucial step in Blelloch scan algorithm; copy the identity to the "end" of the array
	checkCudaErrors( 
//		cudaMemset(&dev_f_in[(1<<Nb)-1], static_cast<T>(0), sizeof(T)) );	
//		cudaMemset(&dev_f_in[L_x-1], static_cast<T>(0), sizeof(T)) );	
		cudaMemset(&(dev_f_in[(1<<Nb)-1]), ((T) 0), sizeof(T)) );	
	

	// do down sweep
	for (unsigned int k = Nb; k>=1; --k) {
		Blelloch_down_global<T><<<N_x,M_in>>>(dev_f_in, dev_f_out, k, L_x) ;
		copy_swap<T><<<N_x,M_in>>>( dev_f_in, dev_f_out, L_x) ; }

/*
	checkCudaErrors( 
		cudaFree( dev_f_out));
	*/
}	


/** ********************************************************************
 ** ************* END OF EXCLUSIVE SCAN, BLELLOCH SCAN ****************/


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE

	const int numBits = 1;
	const int numBins = 1 << numBits; 

	///////////////////////////////////////////////////////////////////
	// grid, block dimensions
	///////////////////////////////////////////////////////////////////
	// M_histo, number of threads in a single thread block for 
	// histo_kernel, in order to "divide up" entire d_inputVals of length numElems
	const int M_histo = 1 << 10;
	const int gridsize_histo = (numElems + M_histo - 1)/M_histo ;
	
	// M_flip, number of threads in a single thread block for flipscan_kernel
	const int M_flip = 1 << 10;
	const int gridsize_flip = (numElems + M_flip - 1)/M_flip ;
		
	// M_scatter, number of threads in a single thread block for scatter_kernel
	const int M_scat = 1 << 10;
	const int gridsize_scat = (numElems + M_scat - 1)/M_scat;	
		
	// END of grid, block dimensions
	///////////////////////////////////////////////////////////////////

		// create and allocate device GPU memory for intermediate values, including histogram for 0,1's
	unsigned int *d_Histo; // d_Histo of size numBins = 2, for 0 and 1
	unsigned int *d_scan ; // d_scan of size numElems

/*
unsigned int *d_out ; // d_scan of size numElems

	checkCudaErrors( 
		cudaMalloc( &d_out, numElems*sizeof(unsigned int) ));
*/

	checkCudaErrors( 
		cudaMalloc( &d_Histo, numBins*sizeof(unsigned int) ));
	checkCudaErrors(
		cudaMemset( d_Histo, 0, numBins*sizeof(unsigned int)) );
	checkCudaErrors( 
		cudaMalloc( &d_scan, numElems*sizeof(unsigned int) ));
	checkCudaErrors(
		cudaMemset( d_scan, 0, numElems*sizeof(unsigned int)) );
		

	// for each bit in each bit position, starting from the LSB (least significant bit)
	for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i++) {
   
		//	1) Histogram of the number of occurrences of each digit
		histo_kernel<<<gridsize_histo,M_histo>>>( d_inputVals, d_Histo, numElems, i) ;


   		//	2) Exclusive Prefix Sum of Histogram
		single_scan_kernel<<<1,numBins, numBins*sizeof(unsigned int)>>>( 
									d_Histo, numBins, numElems);

		//   3) Determine relative offset of each digit
		flipscan_kernel<<<gridsize_flip,M_flip>>>(d_inputVals, d_scan, numElems, i);

		thrust::device_ptr<unsigned int> dev_scan_ptr = thrust::device_pointer_cast(d_scan);

		thrust::exclusive_scan( dev_scan_ptr, dev_scan_ptr + numElems, dev_scan_ptr);

	

//		Blelloch_scan_kernelLauncher<unsigned int, 0>(d_scan, d_out, numElems, 1024);


		//   4) Combine the results of steps 2 & 3 to determine the final
		// 		output location for each element and move it there
		
		scatter_kernel<<<gridsize_scat,M_scat>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos,
											numElems, d_Histo, d_scan,i); 

		// Make sure the final
		// sorted results end up in the output buffer!  Hint: You may need to do a copy
		// at the end.
		checkCudaErrors(
			cudaMemcpy(d_inputVals, d_outputVals, numElems * sizeof(unsigned int) , cudaMemcpyDeviceToDevice));
		checkCudaErrors(
			cudaMemcpy(d_inputPos , d_outputPos , numElems * sizeof(unsigned int) , cudaMemcpyDeviceToDevice));

	}  // END of for loop for each bit from 0, 1, ... 31


	// free device GPU memory
	checkCudaErrors( cudaFree( d_Histo));
	checkCudaErrors( cudaFree( d_scan));

	//checkCudaErrors(cudaFree(d_out));
}
