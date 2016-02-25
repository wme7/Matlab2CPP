
#include "Test.h"

void Manage_Memory(int phase, int tid, float **h_u, float **h_ul, float **d_u, float **d_un){
  cudaError_t Error;
  if (phase==0) {
    // Allocate domain variable on host (master thread)
    *h_u = (float*)malloc((NX)*sizeof(float));
    *h_ul= (float*)malloc((SNX)*sizeof(float));
   }
  if (phase==1) {
    // Allocate subdomain variables on host (All Threads)
    Error = cudaSetDevice(tid);
    if (DEBUG) printf("CUDA error (cudaSetDevice) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_u,(SNX)*sizeof(float));
    if (DEBUG) printf("CUDA error (cudaMalloc d_u) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_un,(SNX)*sizeof(float));
    if (DEBUG) printf("CUDA error (cudaMalloc d_un) in thread %d = %s\n",tid,cudaGetErrorString(Error));
   }
  if (phase==2) {
    // Free the local domain variables (All thread)
    Error = cudaFree(*d_u);
    if (DEBUG) printf("CUDA error (cudaFree d_u) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaFree(*d_un);
    if (DEBUG) printf("CUDA error (cudaFree d_un) in thread %d = %s\n",tid,cudaGetErrorString(Error));
  }
  if (phase==3) {
    // Free the whole domain variables (master thread)
    free(*h_u);
    free(*h_ul);
  }
}

void Manage_Comms(int phase, int tid, float **h_ul, float **d_u){
  cudaError_t Error;
  if (phase==1) {
    // Communicate data from thread to host local domain
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %d) :::\n",phase,tid);
    Error=cudaMemcpy(*h_ul,*d_u,SNX*sizeof(float),cudaMemcpyDeviceToHost); 
    if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
  }
  if (phase==2) {
    // Communicate data from host local domain to thread
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %d) :::\n",phase,tid);
    Error=cudaMemcpy(*d_u,*h_ul,SNX*sizeof(float),cudaMemcpyHostToDevice); 
    if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
  }
}


__global__ void Set_GPU_IC(int tid,float *ut0){
  // Set domain initial condition in local threads
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid==0) {
    ut0[i] = 0.25;
  }
  if (tid>0 && tid<NO_GPU-1) {
    ut0[i] = 0.50;
  }
  if (tid==4) {
    ut0[i] = 0.75;
  }
  
}

void Call_GPU_Init(int tid,float **ut0){
  // Load the initial condition
  int threads = 64;
  int blocks = (SNX + threads - 1)/threads;
  Set_GPU_IC<<<blocks,threads>>>(tid,*ut0);
  if (DEBUG) printf("CUDA error (Set_GPU_IC) in thread %d = %s\n",tid,cudaGetErrorString(cudaPeekAtLastError()));
}

void Update_Domain(int tid, float *u, float *ul){
  // Explicitly copy data arrays
  for (int i = 0; i < SNX; i++) {
    if (i+tid*SNX<NX) { u[i+tid*SNX] = ul[i];}
    else { u[i+tid*SNX] = 1.0;}
  }
}

void Call_Update(int tid, float **u, float **ul){
  // produce explicitly: u=un;
  Update_Domain(tid,*u,*ul);
}

void Save_Results(float *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int i = 0; i < NX; i++) {
      fprintf(pFile, "%d\t %g\n",i,u[i]);
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}
