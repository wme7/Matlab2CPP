
#include "scalarProd.h"

void Manage_Memory(int phase, float **h_a, float **h_b, float **d_a){
  cudaError_t Error;
  if (phase==0) {
    // allocate h_a and h_b
    *h_a = (float*)malloc( N*sizeof(float) );
    *h_b = (float*)malloc( N*sizeof(float) );
  }
  if (phase==1) {
    // allocate d_a
    Error = cudaMalloc((void**)d_a,N*sizeof(float));
    if (DEBUG) printf("CUDA error (cudaMalloc) = %s\n",cudaGetErrorString(Error));
  }
  if (phase==2) {
    // free h_a, h_b and d_a
    free(*h_a);
    free(*h_b);
    cudaFree(*d_a);
  }
}

void Manage_Comms(int phase, float **h_u, float **d_u){
  cudaError_t Error;
  if (phase==0) {
    // send data from host to device
    Error = cudaMemcpy(*d_u,*h_u,N*sizeof(float),cudaMemcpyHostToDevice);
    if (DEBUG) printf("CUDA error (cudaMemcpy d -> h ) = %s\n",cudaGetErrorString(Error)); 
    }
  if (phase==1) {
    // send data from device to host
    Error = cudaMemcpy(*h_u,*d_u,N*sizeof(float),cudaMemcpyDeviceToHost);
    if (DEBUG) printf("CUDA error (cudaMemcpy h -> d ) = %s\n",cudaGetErrorString(Error));
  }
}

__global__ void GPU_Func(float *u){
  // threads index
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  // duplicate the value of vector u
  if (i < N) {
    u[i] = 2*u[i];
  }
}

void My_GPU_Func(float **u){
  int threads= 128;
  int blocks=(N+threads-1)/threads;
  GPU_Func<<<blocks,threads>>>(*u);
}

void Save_Results(float *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int i = 0; i < N; i++) {
 	fprintf(pFile, "%d\t %g\n",i,u[i]);
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}
