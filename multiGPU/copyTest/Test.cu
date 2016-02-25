
#include "Test.h"

void Manage_Memory(int phase, int tid, float **h_u, float **h_ul, float **d_u, float **d_un){
  cudaError_t Error;
  size_t global= (int)NX*sizeof(float);
  size_t local = (int)SNX*sizeof(float);
  if (phase==0) {
    // Allocate domain variable on host (master thread)
    *h_u = (float*)malloc(global);
   }
  if (phase==1) {
    // first allocate the local domains!
    *h_ul = (float*)malloc(local);
    // Allocate subdomain variables on host (All Threads)
    Error = cudaSetDevice(tid);
    if (DEBUG) printf("CUDA error (cudaSetDevice) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_u,local);
    if (DEBUG) printf("CUDA error (cudaMalloc d_u) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_un,local);
    if (DEBUG) printf("CUDA error (cudaMalloc d_un) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaDeviceSynchronize();
    if (DEBUG) printf("CUDA error (Mem. Management Synchronize) in thread %d = %s\n", tid, cudaGetErrorString(Error));
   }
  if (phase==2) {
    // Free the local domain in host threads
    free(*h_ul);
    // Free the local domain in devices
    Error = cudaFree(*d_u);
    if (DEBUG) printf("CUDA error (cudaFree d_u) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaFree(*d_un);
    if (DEBUG) printf("CUDA error (cudaFree d_un) in thread %d = %s\n",tid,cudaGetErrorString(Error));
  }
  if (phase==3) {
    // Free the whole domain variables (master thread)
    free(*h_u);
  }
}

void Manage_Comms(int phase, int tid, float **h_ul, float **d_u){
  cudaError_t Error;
  size_t local = (int)SNX*sizeof(float);
  if (phase==1) {
    // Communicate data from thread to host local domain
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %d) :::\n",phase,tid);
    Error = cudaMemcpy(*h_ul,*d_u,local,cudaMemcpyDeviceToHost); 
    if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
  }
  if (phase==2) {
    // Communicate data from host local domain to thread
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %d) :::\n",phase,tid);
    Error = cudaMemcpy(*d_u,*h_ul,local,cudaMemcpyHostToDevice); 
    if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
  }
}


__global__ void Set_GPU_IC(int tid, float *u0){
  // Set domain initial condition in local threads
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < SNX) {
    if (tid == 0) {
      u0[i] = 0.25; if (i<5) printf("IC data %d, %g\n",i,u0[i]);
    } else if (tid == 1) {
      u0[i] = 0.75; if (i<5) printf("IC data %d, %g\n",i,u0[i]);
    }
  }
}

void Call_GPU_Init(int tid, float **u0){
  // Load the initial condition
  int threads = 128;
  int blocks = (SNX + threads - 1)/threads;
  Set_GPU_IC<<<blocks,threads>>>(tid,*u0);
  if (DEBUG) printf("CUDA error (Set_GPU_IC) in thread %d = %s\n",tid,cudaGetErrorString(cudaPeekAtLastError()));
}

void Update_Domain(int tid, float *u, float *ul){
  // Explicitly copy data arrays
  for (int i = 0; i < SNX; i++) {
    if (i+tid*SNX < NX) { 
      u[i+tid*SNX] = ul[i];
    } else { 
      u[i+tid*SNX] = 1.0;
    }
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
