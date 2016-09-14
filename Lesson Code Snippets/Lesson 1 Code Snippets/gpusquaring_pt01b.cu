/* gpusquaring_pt01b.cu
 * Ernest Yeung 
 * ernestyalumni@gmail.com
 * demonstrate squaring numbers in CUDA C++11/C++14
 * but with timing of code with cudaEvent
 * */
// Compiling tip: I compiled this with
// nvcc -std=c++11 gpusquaring_pt01b.cu
// i.e. with flag -std=c++11

#include <stdio.h>

__global__ void square(float *d_out, float *d_in) {
	int idx = threadIdx.x ;
	float f = d_in[idx]   ;
	d_out[idx] = f * f ;
}

int main(int argc, char **argv) {
	const int ARRAY_SIZE = 64 ;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i) ;
	}
	float h_out[ARRAY_SIZE];
	

	// declare GPU memory pointers
	float *d_in ; 
	float *d_out;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop) ;
	
	// allocate GPU memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_out, ARRAY_BYTES);
	
	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice );

	cudaEventRecord(start);

	// launch the kernel 
	square<<<1, ARRAY_SIZE>>>(d_out, d_in);

	cudaEventRecord(stop);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	// this is to block CPU execution until all previously issued commands on device have completed
	// without this barrier, code would measure kernel launch time and not kernel execution time
	cudaEventSynchronize(stop); 

	float milliseconds = 0 ;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
//	printf("Effective Bandwidth (GB/s): %fn", milliseconds ) ;
	printf("milliseconds it took for kernel execution : %f \n " , milliseconds );
	
	
	// print out the resulting array
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? " \t " : " \n " );
	}
	
	// free GPU memory allocated
	cudaFree(d_in) ;
	cudaFree(d_out);
	
	return 0;
}
