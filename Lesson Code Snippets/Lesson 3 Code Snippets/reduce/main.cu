/* main.cu
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates reduce with parallel and serial implementations,
 * with CUDA C/C++ and global memory and shared memory
 * 
 * */
#include <vector> // std::vector
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
	const int ARRAY_SIZE { (1<<10)+5 } ; /* */
	
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
	
	// declare GPU memory pointers
	float *dev_f_in;
	
	// allocate GPU memory
	checkCudaErrors(
		cudaMalloc((void **) &dev_f_in, ARRAY_SIZE*sizeof(float)) );
	
	// transfer the input array to the GPU
	cudaMemcpy( dev_f_in, host_f_in, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice ) ; 
	
	
	/*
	 * reduce_global
	 *  single run test of reduce_global
	 * */
	float result_reduce; //	*result_reduce_global = 0.f; doesn't work, obtain error and Segmentation Fault
	
	// MANUALLY CHANGE THIS 1: M_x
	int M_x { 16 };
	
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
	cudaMemcpy( dev_f_in, host_f_in, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice ) ; 
	
	// sanity check print out of initial values: 
	std::cout << std::endl << " Initially, " << std::endl;
	for (int i = 0 ; i < DISPLAY_SIZE; ++i)  {
		std::cout << " " << host_f_in[i] ; }
	std::cout << std::endl;
	// END of resetting 
	
	// MANUALLY CHANGE THIS 1: M_x
	M_x =  1024 ;
	
	reduce_shmem(dev_f_in, &result_reduce, ARRAY_SIZE, M_x);
	
	// print out results
	std::cout << " For an array of size " << ARRAY_SIZE << " the result of `reduce_shmem` is " << 
		result_reduce << std::endl ; 


	
	cudaDeviceSynchronize(); 
		
	cudaFree( dev_f_in);
	
	
}
