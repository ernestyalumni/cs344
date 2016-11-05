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

__global__ void global_reduce_kernel( float * d_out, float * d_in) ;

__global__ void shmem_reduce_kernel(float * d_out, const float * d_in);

void reduce_global( float * d_in, float * out, const int L, int M_in); 

void reduce_shmem(float * d_in, float * out, const int L, int M_in) ; 

# endif // _REDUCES_H__
