
#include "heat1d.h"

void Manage_Memory(int phase, float **h_u, float **d_u, float **d_un){
  cudaError_t Error;
  size_t global= NX*sizeof(float);
  if (phase==0) {
    // Allocate domain on host
    *h_u = (float*)malloc(global);
   }
  if (phase==1) {
    // Allocate local domain variable on device
    Error = cudaSetDevice(0);
    if (DEBUG) printf("CUDA error (cudaSetDevice) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_u,global);
    if (DEBUG) printf("CUDA error (cudaMalloc d_u) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_un,global);
    if (DEBUG) printf("CUDA error (cudaMalloc d_un) = %s\n",cudaGetErrorString(Error));
   }
  if (phase==2) {
    // Free local domain variable on device
    Error = cudaFree(*d_u);
    if (DEBUG) printf("CUDA error (cudaFree d_u) = %s\n",cudaGetErrorString(Error));
    Error = cudaFree(*d_un);
    if (DEBUG) printf("CUDA error (cudaFree d_un) = %s\n",cudaGetErrorString(Error));
  }
  if (phase==3) {
    // Free the domain on host
    free(*h_u);
  }
}

void Manage_Comms(int phase, float **h_u, float **d_u){
  cudaError_t Error;
  if (phase==1) {
    // Transfer all data from host to device
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d) :::\n",phase);
    Error=cudaMemcpy(*d_u+1,*h_u+1,(NX-2)*sizeof(float),cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
  }
  if (phase==2) {
    // Transfer all data from device to host
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d) :::\n",phase);
    Error=cudaMemcpy(*h_u+1,*d_u+1,(NX-2)*sizeof(float),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
  }
}

void Set_IC(float *u0){
  // Set initial condition in global domain
  for (int i = 1; i < NX; i++) u0[i]=0.0;
  // Set Dirichlet boundary conditions in global domain
  u0[0]=0.0; u0[NX-1]=1.0;
}

void Call_Init(float **u0){
  // Load the initial condition
  Set_IC(*u0);
}

__global__ void Set_GPU_IC(float *u0){
  // local threads indexes
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // set initial condition only at "interior" nodes
  if (i>0 & i<NX) u0[i]=0.0;
  // Set Dirichlet boundary conditions in global domain
  if (i==  0 ) u0[i]=0.0; 
  if (i==NX-1) u0[i]=1.0;
}

void Call_GPU_Init(float **u0){
  // Load the initial condition
  int threads = 128;
  int blocks = (NX + threads - 1)/threads;
  Set_GPU_IC<<<blocks,threads>>>(*u0);
  if (DEBUG) printf("CUDA error (Set_GPU_IC) = %s\n",cudaGetErrorString(cudaPeekAtLastError()));
}

__global__ void Laplace1d(const float * __restrict__  u, float * __restrict__ un){
  // local threads indexes
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int o =   i  ; // node( j,i ) 
  int r = (i+1); // node(j-1,i)  l--o--r
  int l = (i-1); // node(j,i-1) 

  // only update "interior" nodes
  if(i>0 && i<NX-1) {
    un[o] = u[o] + KX*(u[r]-2*u[o]+u[l]);
  } else {
    un[o] = u[o];
  }
}

void Call_Laplace(float **u, float **un){
  // Produce one iteration of the laplace operator
  int threads = 128;
  int blocks = (NX + threads - 1)/threads;
  Laplace1d<<<blocks,threads>>>(*u,*un);
  if (DEBUG) printf("CUDA error (Call_Laplace) = %s\n",cudaGetErrorString(cudaPeekAtLastError()));
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
