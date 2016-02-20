
#include "dotProd.h"

void Manage_Memory(int phase,float **h_a,float **h_b,float **h_c,float **d_a,float **d_b,float **d_c){
  cudaError_t Error;
  if (phase==0) {
    // allocate h_a, h_b and h_c
    *h_a = (float*)malloc(     N*sizeof(float) );
    *h_b = (float*)malloc(     N*sizeof(float) );
    *h_c = (float*)malloc(BLOCKS*sizeof(float) );
  }
  if (phase==1) {
    // allocate d_a, d_b and h_c
    Error = cudaMalloc((void**)d_a,     N*sizeof(float));
    if (DEBUG) printf("CUDA error (cudaMalloc) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_b,     N*sizeof(float));
    if (DEBUG) printf("CUDA error (cudaMalloc) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_c,BLOCKS*sizeof(float));
    if (DEBUG) printf("CUDA error (cudaMalloc) = %s\n",cudaGetErrorString(Error));
  }
  if (phase==2) {
    // free h_a, h_b and d_a
    free(*h_a);
    free(*h_b);
    free(*h_c);
    cudaFree(*d_a);
    cudaFree(*d_b);
    cudaFree(*d_c);
  }
}

void Manage_Comms(int phase,float **h_u,float **h_v,float **h_w,float **d_u,float **d_v,float **d_w){
  cudaError_t Error;
  if (phase==0) {
    // send data from host to device
    Error = cudaMemcpy(*d_u,*h_u,N*sizeof(float),cudaMemcpyHostToDevice);
    if (DEBUG) printf("CUDA error (cudaMemcpy d -> h ) = %s\n",cudaGetErrorString(Error)); 
    Error = cudaMemcpy(*d_v,*h_v,N*sizeof(float),cudaMemcpyHostToDevice);
    if (DEBUG) printf("CUDA error (cudaMemcpy d -> h ) = %s\n",cudaGetErrorString(Error)); 
    }
  if (phase==1) {
    // send data from device to host
    Error = cudaMemcpy(*h_w,*d_w,BLOCKS*sizeof(float),cudaMemcpyDeviceToHost);
    if (DEBUG) printf("CUDA error (cudaMemcpy h -> d ) = %s\n",cudaGetErrorString(Error));
  }
}

__global__ void GPU_Func(float *u,float *v,float *w){
  __shared__ float cache[THREADS];

  // threads can cache indexes
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cid = threadIdx.x;

  // produce vector product
  if (tid < N) cache[cid] = u[tid]*v[tid];
  
  // synchronize threads
  __syncthreads();

  // for reductions, threadsPerBlock must be a power of 2
  // because of the following code:
  for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
      if (cid < stride) cache[cid] += cache[cid + stride];
      __syncthreads();
  }

  // store result from cache block into w[blockIdx.x]
  if (cid==0) {
    w[blockIdx.x] = cache[0];
  }
}

void My_GPU_Func(float **u,float **v,float **w){
  //int threads= 128;
  //int blocks=(N+threads-1)/threads;
  GPU_Func<<<BLOCKS,THREADS>>>(*u,*v,*w);
}
