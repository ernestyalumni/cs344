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
#include <iostream>

//#include <array>
#include <vector>
#include <algorithm>

#include <thrust/device_ptr.h> // thrust::device_ptr

#include "./histogram/histogram.h" 

#include "./common/checkerror.h" // checkCudaErrors

int main(int argc, char **argv) {

	/** Initialization */
	// i.e. boilerplate
	/** ******************************
	 * MANUALLY change the values for 
	 * numBins_SMALL
	 * numBins
	 * *******************************/

	constexpr const unsigned int numBins_SMALL = 128;
	constexpr const unsigned int numElems_SMALL = (numBins_SMALL-1)*(numBins_SMALL)/2 ;
//	std::array<unsigned int, numElems_SMALL> x_in_SMALL; // test a SMALL histogram; input values or observations 
	std::vector<unsigned int> x_in_SMALL; // test a SMALL histogram; input values or observations 


	constexpr const unsigned int numBins = 1024;
	constexpr const unsigned int numElems = (numBins-1)*(numBins)/2 ;
//	std::array<unsigned int, numElems> x_in; // test a LARGE histogram; input values or observations 
	std::vector<unsigned int> x_in; // test a LARGE histogram; input values or observations 

	// initialize the observation values to construct the targeted histogram
	for (unsigned int i = 0; i < numBins_SMALL; i++) {
		for (unsigned int j = i; j > 0; j--) {
			x_in_SMALL.push_back(i); }
	}
	
	for (unsigned int i = 0; i < numBins; i++) {
		for (unsigned int j = i; j > 0; j--) {
			x_in.push_back(i); }
	}
	
	// sanity check
	std::cout << " numElems_SMALL    : " << numElems_SMALL <<    " numElems    : " << numElems << std::endl;
	std::cout << " x_in_SMALL.size() : " << x_in_SMALL.size() << " x_in.size() : " << x_in.size() << std::endl;

	// more sanity check
	for (auto iter = x_in.begin(); iter < x_in.begin()+42 ; ++iter) {
		std::cout << *iter << " " ; } std::cout << std::endl; 

	for (auto iter = x_in.end() - 42; iter < x_in.end() ; ++iter) {
		std::cout << *iter << " " ; } std::cout << std::endl; 


/*
 * obtained these errors: 
 * /usr/lib/gcc/x86_64-redhat-linux/5.3.1/include/mwaitxintrin.h(36): error: identifier "__builtin_ia32_monitorx" is undefined

/usr/lib/gcc/x86_64-redhat-linux/5.3.1/include/mwaitxintrin.h(42): error: identifier "__builtin_ia32_mwaitx" is undefined

2 errors detected in the compilation of "/tmp/tmpxft_000028ec_00000000-7_main.cpp1.ii".

	std::vector<unsigned int>::iterator max_x_in_SMALL = std::max_element( x_in_SMALL.begin(), x_in_SMALL.end()) ; 

/*	std::cout << " max element of x_in_SMALL : " << 
		&max_x_in_SMALL << std::endl; 

	The solution is to add this flag to the compiler makefile

	-D_MWAITXINTRIN_H_INCLUDED

	cf. https://github.com/tensorflow/tensorflow/issues/1066

	*/
	
	std::cout << " max element of x_in_SMALL : " << 
		*(std::max_element( x_in_SMALL.begin(), x_in_SMALL.end() ) ) << std::endl;
	
	// use std::random_shuffle to permute order of the observations 
	std::random_shuffle( x_in_SMALL.begin(), x_in_SMALL.end() ) ; 
	std::random_shuffle( x_in.begin(), x_in.end() ) ; 

	std::cout << " After std::random_shuffle, this is x_in : " << std::endl;
	for (auto iter = x_in.begin(); iter < x_in.begin()+42 ; ++iter) {
		std::cout << *iter << " " ; } std::cout << std::endl; 
	// END of sanity checks

	/**
	 * ************** GPU boilerplate *****************
	 * */
	/**
	 * *********** block, grid dimensions *************
	 * */
	// 1st. phase - input values from global memory into shared memory local histograms
	// notice that gridsize1 will be necessary for size of PartialHisto
//	const unsigned int block_g = 64;
//	const unsigned int gridsize1  = (numElems + blocksize1 - 1)/blocksize1 ; 
	
	// 2nd. phase - merge, concatenate, accumulate - block, grid dimensions
//	const unsigned int block_accum = 64 ; 
//	const int grid_accum = ( numBins + block_accum - 1)/block_accum;
	
	// histogram_global
	const unsigned int blocksize_global_SMALL = 32;
	const unsigned int gridsize_global_SMALL  = (numElems_SMALL + blocksize_global_SMALL - 1)/blocksize_global_SMALL ; 

	const unsigned int blocksize_global = 128;
	const unsigned int gridsize_global  = (numElems + blocksize_global - 1)/blocksize_global ; 
	
	// histogram_shared
	const unsigned int blocksize_s_SMALL = 128; // numBins_SMALL = 128
	const unsigned int gridsize_s_SMALL  = (numElems_SMALL + blocksize_global_SMALL - 1)/blocksize_global_SMALL ; 

	const unsigned int blocksize_s = 1024;  // numBins  = 1024
	const unsigned int gridsize_s  = (numElems + blocksize_global - 1)/blocksize_global ; 

	// histogram_sharedatomics
	// 1st. phase - input values from global memory into shared memory local histograms
	// notice that gridsize1 will be necessary for size of PartialHisto
	const unsigned int blocksize1_SMALL = 32;
//	const unsigned int gridsize1_SMALL  = (numElems + blocksize1_SMALL - 1)/blocksize1_SMALL ; 

	const unsigned int blocksize1 = 64;
//	const unsigned int gridsize1  = (numElems + blocksize1 - 1)/blocksize1 ; 
	
	// 2nd. phase - merge, concatenate, accumulate - block, grid dimensions
	const unsigned int block_accum_SMALL = 32 ; 
	const unsigned int block_accum = 64 ; 
//	const int grid_accum = ( numBins + block_accum - 1)/block_accum;
	
	
	/** END of block, grid dimensions *****************/
	// END of block, grid dimensions


	// create and allocate device GPU memory for input values, and then initialize with host input values
	unsigned int *d_vals_SMALL;
	checkCudaErrors(cudaMalloc(&d_vals_SMALL, sizeof(unsigned int) * numElems_SMALL) );
	checkCudaErrors(cudaMemcpy(d_vals_SMALL, x_in_SMALL.data(), sizeof(unsigned int) * numElems_SMALL, cudaMemcpyHostToDevice) );

	unsigned int *d_vals;
	checkCudaErrors(cudaMalloc(&d_vals, sizeof(unsigned int) * numElems) );
	checkCudaErrors(cudaMemcpy(d_vals, x_in.data(), sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice) );

	// wrap raw pointer with a device_ptr
	thrust::device_ptr<unsigned int> dev_vals_ptr_SMALL = thrust::device_pointer_cast( d_vals_SMALL);
	thrust::device_ptr<unsigned int> dev_vals_ptr = thrust::device_pointer_cast( d_vals);
	

	// sanity check if our dev_vals_ptr points to device d_vals and had successfully copied values
	std::cout << "\n  check if x_in copied to d_vals and dev_vals_ptr points to it : " << std::endl ; 
	for (auto iter = dev_vals_ptr; iter < dev_vals_ptr + 42; iter++) {
		std::cout << *iter << " " ;
	} std::cout << std::endl; 

	// create and allocate device GPU memory for (final) histograms
	unsigned int *d_Histo_SMALL ;
	checkCudaErrors(cudaMalloc(&d_Histo_SMALL, sizeof(unsigned int)*numBins_SMALL ) );
	checkCudaErrors(cudaMemset(d_Histo_SMALL, 0, sizeof(unsigned int)*numBins_SMALL  ) );

	unsigned int *d_Histo ;
	checkCudaErrors(cudaMalloc(&d_Histo, sizeof(unsigned int)*numBins ) );
	checkCudaErrors(cudaMemset(d_Histo, 0, sizeof(unsigned int)*numBins  ) );

	// wrap raw pointer with a device_ptr
	thrust::device_ptr<unsigned int> dev_Histo_ptr_SMALL = thrust::device_pointer_cast( d_Histo_SMALL);
	thrust::device_ptr<unsigned int> dev_Histo_ptr = thrust::device_pointer_cast( d_Histo);


	// sanity check if dev_Histo_ptr points to device d_Histo and had successfully set values to 0
	std::cout << "\n  check if d_Histo set to 0 and dev_Histo_ptr points to it : " << std::endl ; 
	for (auto iter = dev_Histo_ptr; iter < dev_Histo_ptr + 42; iter++) {
		std::cout << *iter << " " ;
	} std::cout << " \n " << std::endl; 
	
	
	// END GPU boilerplate
	// END boilerplate/Initialization
	
	/************************
	 * serial, CPU histogram
	 ************************/ 
	std::vector<unsigned int> Histo_CPU_SMALL(numBins_SMALL); 
	for (unsigned int i = 0; i < numElems_SMALL; ++i) { 
		Histo_CPU_SMALL[ x_in_SMALL[i] ]++ ; 
	}
	
		// sanity check for Histo_CPU
	std::cout << " Histogram on the CPU for SMALL size : " << std::endl ;
	for (unsigned int i = numBins_SMALL/2 - 23 ; i < numBins_SMALL/2 + 23; ++i) {
		std::cout << Histo_CPU_SMALL[ i ] << " " ; }  std::cout << std::endl;

	std::vector<unsigned int> Histo_CPU(numBins); 
	for (unsigned int i = 0; i < numElems; ++i) { 
		Histo_CPU[ x_in[i] ]++ ; 
	}
	
		// sanity check for Histo_CPU
	std::cout << " Histogram on the CPU for large size : " << std::endl ;
	for (unsigned int i = 0; i < 42; ++i) {
		std::cout << Histo_CPU[ i ] << " " ; }  std::cout << std::endl;

	/************************
	 * global, GPU histogram
	 ************************/ 

	histogram_global<<<gridsize_global_SMALL, blocksize_global_SMALL>>>(d_vals_SMALL, 
		d_Histo_SMALL, numBins_SMALL, numElems_SMALL) ;
	histogram_global<<<gridsize_global, blocksize_global>>>(d_vals, d_Histo, numBins, numElems) ;

		// sanity check
	std::cout << "\n histogram_global : " << std::endl;
	for (auto iter = dev_Histo_ptr; iter < dev_Histo_ptr + 42; iter++) {
		std::cout << *iter << " " ;
	} std::cout << " \n " << std::endl; 
	
	/************************
	 * shared memory, GPU histogram
	 ************************/ 
	// RESET the values of d_Histo_SMALL, d_Histo
	checkCudaErrors(cudaMemset(d_Histo_SMALL, 0, sizeof(unsigned int)*numBins_SMALL  ) );
	checkCudaErrors(cudaMemset(d_Histo, 0, sizeof(unsigned int)*numBins  ) );

	// sanity check if dev_Histo_ptr points to device d_Histo and had successfully set values to 0
	std::cout << "\n  check if d_Histo set to 0 and dev_Histo_ptr points to it : " << std::endl ; 
	for (auto iter = dev_Histo_ptr; iter < dev_Histo_ptr + 42; iter++) {
		std::cout << *iter << " " ;
	} std::cout << " \n " << std::endl; 

	// sanity check if our dev_vals_ptr still to device d_vals with successfully copied values
	std::cout << "\n  check if x_in copied to d_vals and dev_vals_ptr points to it : " << std::endl ; 
	for (auto iter = dev_vals_ptr; iter < dev_vals_ptr + 42; iter++) {
		std::cout << *iter << " " ;
	} std::cout << std::endl; 



	histogram_shared<<<gridsize_s_SMALL, blocksize_s_SMALL, 
		sizeof(unsigned int) * numBins_SMALL >>>(d_vals_SMALL, 
		d_Histo_SMALL, numBins_SMALL, numElems_SMALL) ;

	histogram_shared<<<gridsize_s, blocksize_s, 
		sizeof(unsigned int) * numBins >>>(d_vals, d_Histo, numBins, numElems) ;

		// sanity check
	std::cout << "\n histogram_shared : " << std::endl;
	for (auto iter = dev_Histo_ptr; iter < dev_Histo_ptr + 42; iter++) {
		std::cout << *iter << " " ;
	} std::cout << " \n " << std::endl; 
	for (auto iter = dev_Histo_ptr + numBins - 42; iter < dev_Histo_ptr + numBins; iter++) {
		std::cout << *iter << " " ;
	} std::cout << " \n " << std::endl; 

	/****************************************
	 * shared atomics memory, GPU histogram
	 ***************************************/ 
	// RESET the values of d_Histo_SMALL, d_Histo
	checkCudaErrors(cudaMemset(d_Histo_SMALL, 0, sizeof(unsigned int)*numBins_SMALL  ) );
	checkCudaErrors(cudaMemset(d_Histo, 0, sizeof(unsigned int)*numBins  ) );

	// sanity check if dev_Histo_ptr points to device d_Histo and had successfully set values to 0
	std::cout << "\n  check if d_Histo set to 0 and dev_Histo_ptr points to it : " << std::endl ; 
	for (auto iter = dev_Histo_ptr; iter < dev_Histo_ptr + 42; iter++) {
		std::cout << *iter << " " ;
	} std::cout << " \n " << std::endl; 


/*	sanity check of first phase
 * 
	unsigned int gridSize = (numElems + blocksize1 - 1)/blocksize1 ; 
	unsigned int * d_PartialHisto ; // Partial Histograms, of size numBins * gridSize 
	checkCudaErrors( cudaMalloc( &d_PartialHisto, sizeof(unsigned int) * numBins * gridSize ) ) ;
	checkCudaErrors( cudaMemset( d_PartialHisto, 0, sizeof(unsigned int) * numBins * gridSize));

	histogram_smem_atomics<<<gridSize,blocksize1, numBins * sizeof(unsigned int )>>>(d_vals, d_PartialHisto, numBins, numElems) ;


	checkCudaErrors( cudaFree( d_PartialHisto ));
	*/
	
	histogram_shared_atomics_kernel( d_vals, d_Histo, 
		numBins, numElems, blocksize1, block_accum);
	
		// sanity check
	std::cout << "\n histogram_smem_atomics : " << std::endl;
	for (auto iter = dev_Histo_ptr; iter < dev_Histo_ptr + 42; iter++) {
		std::cout << *iter << " " ;
	} std::cout << " \n " << std::endl; 
	for (auto iter = dev_Histo_ptr + numBins - 42; iter < dev_Histo_ptr + numBins; iter++) {
		std::cout << *iter << " " ;
	} std::cout << " \n " << std::endl; 





		// free up GPU memory
	checkCudaErrors( cudaFree( d_vals_SMALL) ) ;
	checkCudaErrors( cudaFree( d_vals) ) ;
	checkCudaErrors( cudaFree( d_Histo_SMALL ) ) ;
	checkCudaErrors( cudaFree( d_Histo ) ) ;

	return 0;
}

 
 
