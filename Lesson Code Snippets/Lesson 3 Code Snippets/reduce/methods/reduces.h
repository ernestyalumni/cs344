/* reduces.h
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates reduce with parallel and serial implementations, 
 * with CUDA C/C++ and global and shared memories
 * 
 * */
#ifndef __REDUCES_H__
#define __REDUCES_H__

#include <vector>
#include "checkerror.h" // checkCudaErrors

// parallel implementations

__global__ void global_reduce_kernel( float * d_in, float * d_out, const int L) ;

__global__ void shmem_reduce_kernel(const float * d_in, float * d_out );

__global__ void shmem_reduce_max_kernel( const float * d_in, float * d_out) ;

__global__ void shmem_reduce_min_kernel( const float * d_in, float * d_out) ;

void reduce_global( float * d_in, float * out, const int L, int M_in); 

void reduce_shmem(float * d_in, float * out, const int L, int M_in) ; 

void reduce_shmem_max(float * d_in, float * out, const int L, int M_in) ; 

void reduce_shmem_min(float * d_in, float * out, const int L, int M_in) ; 


#endif // __REDUCES_H__
