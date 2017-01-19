/* main.cu
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates reduce with parallel and serial implementations,
 * with CUDA C/C++ and global memory and shared memory
 * 
 * Compilation tip:
 * nvcc -std=c++11 -D_MWAITXINTRIN_H_INCLUDED main.cu -dc -o main.o
 * 
 * 
 * */
#include <iomanip>  // std::setprecision
#include <vector>   // std::vector
/*
 * Note that #include <algorithm> with nvcc requires another flag, otherwise this error is obtained:
 * /usr/lib/gcc/x86_64-redhat-linux/5.3.1/include/mwaitxintrin.h(36): error: identifier "__builtin_ia32_monitorx" is undefined
 * add this:
 * 	-D_MWAITXINTRIN_H_INCLUDED

 * */
#include <algorithm> // std::for_each

#include <chrono> // chrono::steady_clock::now()

#include "./methods/checkerror.h"
#include "./methods/reduces.h" /* global_reduce_kernel, shmem_reduce_kernel, 
								* reduce_global 
								* */

int main() {
	// "boilerplate"
	// initiate correct GPU
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		exit(EXIT_FAILURE);
	}
	int dev = 0;
	cudaSetDevice(dev);
	
	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0) {
		std::cout << " Using device " << dev << ":\n" ;
		std::cout << devProps.name << "; global mem: " << (int)devProps.totalGlobalMem <<
			"; compute v" << (int)devProps.major << "." << (int)devProps.minor << "; clock: " <<
			(int)devProps.clockRate << " kHz" << std::endl; }
	// END if GPU properties
	
	// MANUALLY CHANGE THESE 2: ARRAY_SIZE, DISPLAY_SIZE
	// input array with interesting values "boilerplate"
	const int ARRAY_SIZE { (1<<8)+0 } ; /* */
	
	std::cout << "For an (float) array of size (length) : " << ARRAY_SIZE << std::endl ;
	std::cout << "or, in bytes, " << ARRAY_SIZE*sizeof(float) << std::endl;
	
	const int DISPLAY_SIZE = 22; 		// how many numbers to display, read out, i.e. print out on screen
	static_assert( ARRAY_SIZE >= DISPLAY_SIZE, "ARRAY_SIZE needs to be equal or bigger than DISPLAY_SIZE");
	
	// generate input array on host
	std::vector<float> f_vec;
	for (auto i = 0; i < ARRAY_SIZE; ++i) {
		f_vec.push_back(i) ; }
	float* host_f_in;
	host_f_in = f_vec.data();
	
	// sanity check print out of initial values: 
	std::cout << " Initially, " << std::endl;
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << f_vec[i] ; }
	std::cout << std::endl;
	// END of initializing, creating input array with interesting values, on host CPU, boilerplate
	/**
	 *******************************************************************
	 * Further BOILERPLATE
	 ******************************************************************/
	 // Initialization (on host CPU)
	std::vector<float> x(1<<20); // test a LARGE histogram; input values or observations 
	
// cf. http://stackoverflow.com/questions/3752019/how-to-get-the-index-of-a-value-in-a-vector-using-for-each
	int j = 0;
//	std::for_each(x.begin(), x.end(), [&j](float const& value) { j++; });
	std::for_each(x.begin(), x.end(), [&j](float &value) { value = j++; });
	std::cout << j << std::endl;

	// sanity check print out of initial values: 
	std::cout << std::endl << " Initially, (for x)" << std::endl;
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << x[i] ; }
	std::cout << std::endl;

	// cf. http://en.cppreference.com/w/cpp/algorithm/for_each
	std::for_each(x.begin(), x.end(), [](float &n){ n++; });
	std::cout << std::endl << " After for_each with [](int &n){ n++ }, " << std::endl;
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << x[i] ; }
	for (int i = x.size() - DISPLAY_SIZE ; i < x.size() ; ++i)  {
		std::cout << " " << std::setprecision(7) << x[i] ; }
	std::cout << std::endl;
	std::cout << " max element of x : " << 
		*(std::max_element( x.begin(), x.end() ) ) << std::endl;

	std::vector<float>::const_iterator it = x.begin();
//	std::vector<float>::iterator it = x.begin();  // same result
//	std::for_each(x.begin(), x.end(), [&it](float const& value) { ++it ; } );
	std::for_each(x.begin(), x.end(), [&it](float & value) { value = *(++it) ; } );
	
	std::cout << std::endl << " After for_each with [](float const & value){ ++it }, " << std::endl;
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << x[i] ; }
	for (int i = x.size() - DISPLAY_SIZE ; i < x.size() ; ++i)  {
		std::cout << " " << std::setprecision(7) << x[i] ; }
	std::cout << std::endl;
	std::cout << " max element of x : " << 
		*(std::max_element( x.begin(), x.end() ) ) << std::endl;
	// END of sanity check 

	// use std::random_shuffle to permute order of the elements 
	std::random_shuffle( x.begin(), x.end() ) ; 
		
		// sanity check
	std::cout << std::endl << " After std::random_shuffle, " << std::endl;
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << x[i] ; }
	for (int i = x.size() - DISPLAY_SIZE ; i < x.size() ; ++i)  {
		std::cout << " " << std::setprecision(7) << x[i] ; }
	std::cout << std::endl;
	std::cout << " max element of x : " << 
		*(std::max_element( x.begin(), x.end() ) ) << std::endl;
		// END of sanity check 

/** ********************************************************************
 * device GPU boilerplate
 * */
	
	// declare GPU memory pointers
	float *dev_f_in;
	
	// allocate GPU memory
	checkCudaErrors(
		cudaMalloc((void **) &dev_f_in, ARRAY_SIZE*sizeof(float)) );
	
	// transfer the input array to the GPU
	cudaMemcpy( dev_f_in, host_f_in, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice ) ; 

	
	// declare GPU memory pointers
	float *d_x;
	
	// allocate GPU memory
	checkCudaErrors(
		cudaMalloc((void **) &d_x, x.size()*sizeof(float)) );
	
	// transfer the input array to the GPU
	cudaMemcpy( d_x, x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice ) ; 

/** ******************************************************************** 
 * END of device GPU boilerplate
 * */


	
	/*
	 * reduce_global
	 *  single run test of reduce_global
	 * */
	float result_reduce; //	*result_reduce_global = 0.f; doesn't work, obtain error and Segmentation Fault
	
	// MANUALLY CHANGE THIS 1: M_x
	int M_x { 1024 };
	
	reduce_global(dev_f_in, &result_reduce, ARRAY_SIZE, M_x);
	
	// print out results
	std::cout << " For an array of size " << ARRAY_SIZE << " the result of `reduce_global` is " << 
		result_reduce << std::endl ; 


	/*
	 * reduce_shmem
	 *  single run test of reduce_global
	 * */
	// reset result_reduce and dev_f_in
	result_reduce = 0.f ; 
	std::cout << " Reset result_reduce : " << result_reduce << std::endl;
	checkCudaErrors( 
		cudaMemcpy( dev_f_in, host_f_in, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice ) ); 
	
	// sanity check print out of initial values: 
	std::cout << std::endl << " Initially, " << std::endl;
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << host_f_in[i] ; }
	std::cout << std::endl;
	// END of resetting 
	
	// MANUALLY CHANGE THIS 1: M_x
	M_x =  32 ;
	
	reduce_shmem(dev_f_in, & result_reduce, ARRAY_SIZE, M_x);
	
	// print out results
	std::cout << " For an array of size " << ARRAY_SIZE << " the result of `reduce_shmem` is " << 
		result_reduce << std::endl ; 

	/*******************************************************************
	 * reduces on x, d_x
	 * ****************************************************************/
	std::cout << "\n for a large array x of size " << x.size() << " : " << std::endl ; 
	reduce_shmem(d_x, &result_reduce, x.size(), 1024);

	// print out results
	std::cout << " For an array of size " << x.size() << " the result of `reduce_shmem` is " << 
		std::setprecision(12) << result_reduce << std::endl ; 

	// sanity check
	/*
	checkCudaErrors( 
		cudaMemcpy( x.data(), d_x, x.size()*sizeof(float), cudaMemcpyDeviceToHost) );
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << x[i] ; }
	std::cout << std::endl;
*/


	// find the max
	reduce_shmem_max(d_x, &result_reduce, x.size(), 1024);

	// print out results
	std::cout << " For an array of size " << x.size() << " the result of `reduce_shmem_max` is " << 
		std::setprecision(12) << result_reduce << std::endl ; 

	// find the min
	reduce_shmem_min(d_x, &result_reduce, x.size(), 1024);

	// print out results
	std::cout << " For an array of size " << x.size() << " the result of `reduce_shmem_min` is " << 
		std::setprecision(12) << result_reduce << std::endl ; 



	/*******************************************************************
	 * END reduces on x, d_x
	 * ****************************************************************/
	
	cudaDeviceSynchronize(); 
		
	checkCudaErrors( cudaFree( dev_f_in) );
	checkCudaErrors( cudaFree( d_x ) );

	
}
