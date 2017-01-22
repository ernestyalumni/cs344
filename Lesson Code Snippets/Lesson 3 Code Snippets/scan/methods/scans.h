/* scans.h
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates Hillis/Steele and Blelloch (exclusive) scan with a parallel implementation
 * with CUDA C/C++ and global memory
 * 
 * */
#ifndef __SCANS_H__
#define __SCANS_H__

#include <vector>
#include "checkerror.h" // checkCudaErrors

// parallel implementations

__global__ void Blelloch_up_global(float* f_in, float* f_out, const int k, const int L_x); 

__global__ void Blelloch_down_global( float* f_in, float* f_out, const int k, const int L_x); 

__global__ void copy_swap(float* f_in, float* f_target, const int L_x) ;

void Blelloch_scan_kernelLauncher(float* dev_f_in, float* dev_f_out, const int L_x, const int M_in); 

__global__ void HillisSteele_global(float* f_in, float* f_out, const int k, const int L_x); 

void HillisSteele_kernelLauncher(float* dev_f_in, float* dev_f_out, const int L_x, const int M_in) ;


	// Blelloch (exclusive) scan 
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
void Blelloch_scan_kernelLauncher(T* dev_f_in, const unsigned int L_x, 
									const unsigned int M_in) {
//	auto Nb = static_cast<int>(std::log2( L_x) );
	unsigned int Nb  = ((unsigned int) log2( ((double) L_x)) );
	
	T* dev_f_out;
	checkCudaErrors( 
		cudaMalloc( &dev_f_out, L_x*sizeof(T) ));
	checkCudaErrors(
		cudaMemset( dev_f_out, 0, L_x*sizeof(T) ));

	// determine number of thread blocks to be launched
	const unsigned int N_x  =  ( L_x + M_in - 1 ) / M_in  ;

	// do up sweep
	for (unsigned int k = 1; k <= Nb; ++k) {
		Blelloch_up_global<T><<<N_x, M_in>>>( dev_f_in, dev_f_out, k, L_x) ; 
		copy_swap<T><<<N_x,M_in>>>(dev_f_in, dev_f_out,L_x); }
		
	// crucial step in Blelloch scan algorithm; copy the identity to the "end" of the array
	checkCudaErrors( 
//		cudaMemset(&dev_f_in[(1<<Nb)-1], static_cast<T>(0), sizeof(T)) );	
		cudaMemset(&dev_f_in[L_x-1], static_cast<T>(0), sizeof(T)) );	
	

	// do down sweep
	for (unsigned int k = Nb; k>=1; --k) {
		Blelloch_down_global<T><<<N_x,M_in>>>(dev_f_in, dev_f_out, k, L_x) ;
		copy_swap<T><<<N_x,M_in>>>( dev_f_in, dev_f_out, L_x) ; }

	checkCudaErrors( 
		cudaFree( dev_f_out));
	

}	




// serial implementation

void blelloch_up( std::vector<float> f_in, std::vector<float> &f_out, const int k ) ;

void blelloch_down( std::vector<float> f_in, std::vector<float> &f_out, const int k); 

//void blelloch_serial( std::vector<float> &f_in, std::vector<float> &f_out, const int N);

void blelloch_serial( std::vector<float>& f_in ) ;

void HillisSteele_serial( std::vector<float>& f_in ) ;



#endif // __SCANS_H__
