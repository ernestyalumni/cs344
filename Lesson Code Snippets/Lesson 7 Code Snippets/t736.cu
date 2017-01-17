/*
 * IMPORTANT: compilation tip
 * 20170116 I haven't made a hard "symbolic" link to the ROOT directory for what libraries to automatically include
 * i.e. that /usr/lib thing
 * so here's what I do, for right now
 * nvcc -I./cub-1.6.4/ t736.cu -o t736
 * 
 * */
#include <cub/cub.cuh>
#include <stdio.h>
typedef int mytype;
int main(){
  // Declare, allocate, and initialize device pointers for input and output
  size_t num_items = 35000000;
  mytype *d_in;
  mytype *h_in;
  mytype *d_out;
  size_t sz = num_items*sizeof(mytype);
  h_in = (mytype *)malloc(sz);
  if (!h_in) {printf("malloc fail\n"); return -1;}
  cudaMalloc(&d_in,  sz);
  cudaMalloc(&d_out, sz);
  for (size_t i = 0; i < num_items; i++) h_in[i] = 1;
  cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);
  printf("\nInput:\n");
  for (int i = 0; i < 10; i++) printf("%d ", h_in[i]);
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
// 
  cudaMemcpy(h_in, d_out, sz, cudaMemcpyDeviceToHost);
  printf("\nOutput:\n");
  for (int i = 0; i < 10; i++) printf("%d ", h_in[i]);
  printf("\n");
  return 0;
}
