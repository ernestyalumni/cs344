/** 
 * histogram_compare.cu
 * 
 * \file histogram_compare.cu
 * \author typed up by Ernest Yeung
 * \brief comparing histogram methods
 * 
 * 
 * typed up by Ernest Yeung  ernestyalumni@gmail.com
 * \date 20170110
 * cf. http://www.orangeowlsolutions.com/archives/1178
 * 
 * Also in 
 * cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming 
 * Chapter 9 Atomics 
 * 9.4 Computing Histograms
 * 9.4.1 CPU Histogram Computation
 * and 
 * cf. https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/histshared.cu
 * 
 * Compilation tip
 * nvcc -std=c++11 histogram_compare.cu -o histogram_compare.exe
 *  
 * 
 */
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>

#include "utils.h" // checkCudaErrors()

#define SIZE (100*1024*1024) // 100 MB

/**********************************************/
/* FUNCTION TO GENERATE RANDOM UNSIGNED CHARS */
/**********************************************/

unsigned char* big_random_block(int size) {
	unsigned char *data = (unsigned char*)malloc(size);
	for (int i=0; i<size; i++) 
		data[i] = rand();
	return data;
}
// GPU with atomics; i.e.
// GPU with (global) atomics
/****************************************/
/* GPU HISTOGRAM CALCULATION VERSION 1  */
/****************************************/
__global__ void histo_kernel1(unsigned char *buffer, long size, unsigned int *histo) {
	
	// --- The number of threads does not cover all the data size
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (i < size) {
		atomicAdd(&histo[buffer[i]], 1);
		i += stride;
	}
}

// GPU with atomics in shared memory with final summation of partial histograms
/****************************************/
/* GPU HISTOGRAM CALCULATION VERSION 2  */
/****************************************/
__global__ void histo_kernel2(unsigned char *buffer, long size, unsigned int *histo) {
	
	// --- Allocating and initializing shared memory to store partial histograms
	__shared__ unsigned int temp[256];
	temp[threadIdx.x] = 0;
	__syncthreads();
	
	// --- The number of threads does not cover all the data size
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x; 
	while (i < size)
	{
		atomicAdd(&temp[buffer[i]], 1);
		i += offset;
	}
	__syncthreads();
	
	// --- Summing histograms
	atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

/********/
/* MAIN */
/********/
int main() {
	
	// --- Generating an array of SIZE unsigned chars
	unsigned char *buffer = (unsigned char*)big_random_block(SIZE);
	
	/********************/
	/* CPU COMPUTATIONS */
	/********************/
	
	// --- Allocating host memory space and initializing the host-side histogram
	unsigned int histo[256];
	for (int i = 0; i < 256; i++) histo[i] = 0;
	
	clock_t start_CPU, stop_CPU;
	
	// --- Histogram calculation on the host
	start_CPU = clock();
	for (int i=0; i<SIZE; i++) histo [buffer[i]]++;
	stop_CPU = clock();
	float elapsedTime = (float)(stop_CPU - start_CPU) / (float) CLOCKS_PER_SEC * 1000.0f ;
	
	 printf("Time to generate (CPU): %3.1f ms \n", elapsedTime);

    // --- Indirect check of the result
    long histoCount = 0;
    for (int i=0; i<256; i++) { histoCount += histo[i]; }
    printf("Histogram Sum: %ld \n", histoCount);

    /********************/
    /* GPU COMPUTATIONS */
    /********************/

    // --- Initializing the device-side data
    unsigned char *dev_buffer;
    checkCudaErrors(cudaMalloc((void**)&dev_buffer,SIZE));
    checkCudaErrors(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));

    // --- Allocating device memory space for the device-side histogram
    unsigned int *dev_histo;
    checkCudaErrors(cudaMalloc((void**)&dev_histo,256*sizeof(long)));

    // --- GPU timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // --- ATOMICS
    // --- Histogram calculation on the device - 2x the number of multiprocessors gives best timing
    checkCudaErrors(cudaEventRecord(start,0));
    checkCudaErrors(cudaMemset(dev_histo,0,256*sizeof(int)));
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop,0));
    int blocks = prop.multiProcessorCount;
    histo_kernel1<<<blocks*2,256>>>(dev_buffer, SIZE, dev_histo);

    checkCudaErrors(cudaMemcpy(histo,dev_histo,256*sizeof(int),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop,0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime,start,stop));
    printf("Time to generate (GPU):  %3.1f ms \n", elapsedTime);

    histoCount = 0;
    for (int i=0; i<256; i++) {
        histoCount += histo[i];
    }
    printf( "Histogram Sum:  %ld \n", histoCount );

    // --- Check the correctness of the results via the host
    for (int i=0; i<SIZE; i++) histo[buffer[i]]--;
    for (int i=0; i<256; i++) {
        if (histo[i] != 0) printf( "Failure at %d!  Off by %d \n", i, histo[i] );
}

    // --- ATOMICS IN SHARED MEMORY
    // --- Histogram calculation on the device - 2x the number of multiprocessors gives best timing
    checkCudaErrors(cudaEventRecord(start,0));
    checkCudaErrors(cudaMemset(dev_histo,0,256*sizeof(int)));
    checkCudaErrors(cudaGetDeviceProperties(&prop,0));
    blocks = prop.multiProcessorCount;
    histo_kernel2<<<blocks*2,256>>>(dev_buffer, SIZE, dev_histo);

    checkCudaErrors(cudaMemcpy(histo,dev_histo,256*sizeof(int),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop,0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime,start,stop));
    printf("Time to generate (GPU):  %3.1f \n", elapsedTime);

    histoCount = 0;
    for (int i=0; i<256; i++) {
        histoCount += histo[i];
    }
    printf( "Histogram Sum:  %ld \n", histoCount );

    // --- Check the correctness of the results via the host
    for (int i=0; i<SIZE; i++) histo[buffer[i]]--;
    for (int i=0; i<256; i++) {
        if (histo[i] != 0) printf( "Failure at %d!  Off by %d \n", i, histo[i] );
    }

    // --- CUDA THRUST

    checkCudaErrors(cudaEventRecord(start,0));

    // --- Wrapping dev_buffer raw pointer with a device_ptr and initializing a device_vector with it
    thrust::device_ptr<unsigned char> dev_ptr(dev_buffer);
    thrust::device_vector<unsigned char> dev_buffer_thrust(dev_ptr, dev_ptr + SIZE);

    // --- Sorting data to bring equal elements together
    thrust::sort(dev_buffer_thrust.begin(), dev_buffer_thrust.end());

    // - The number of histogram bins is equal to the maximum value plus one
    int num_bins = dev_buffer_thrust.back() + 1;

    // --- Resize histogram storage
    thrust::device_vector<int> d_histogram;
    d_histogram.resize(num_bins);

    // --- Find the end of each bin of values
    thrust::counting_iterator<int> search_begin(0);
    thrust::upper_bound(dev_buffer_thrust.begin(), dev_buffer_thrust.end(),
                    search_begin, search_begin + num_bins,
                    d_histogram.begin());

    // --- Compute the histogram by taking differences of the cumulative histogram
    thrust::adjacent_difference(d_histogram.begin(), d_histogram.end(),
                            d_histogram.begin());

    thrust::host_vector<int> h_histogram(d_histogram);
    checkCudaErrors(cudaEventRecord(stop,0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime,start,stop));
    printf("Time to generate (GPU):  %3.1f \n", elapsedTime);

    histoCount = 0;
    for (int i=0; i<256; i++) {
        histoCount += h_histogram[i];
    }
    printf( "Histogram Sum:  %ld \n", histoCount );

    // --- Check the correctness of the results via the host
    for (int i=0; i<SIZE; i++) h_histogram[buffer[i]]--;
    for (int i=0; i<256; i++) {
        if (h_histogram[i] != 0) printf( "Failure at %d!  Off by %d \n", i, h_histogram[i] );
    }

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(dev_histo));
    checkCudaErrors(cudaFree(dev_buffer));

    free(buffer);

    getchar();

}
	
	
	
	
	 
