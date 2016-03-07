
#include "heat2d.h"

void Manage_Memory(int phase, int tid, float **h_u, float **d_u, float **d_un){
  cudaError_t Error;
  size_t global= ( NX+2)*sizeof(float);
  size_t local = (SNX+2)*sizeof(float);
  if (phase==0) {
    // Allocate domain on host
    *h_u = (float*)malloc(global);
   }
  if (phase==1) {
    // Allocate local domain variable on host
    //*h_ul = (float*)malloc(local);
    // Allocate local domain variable on device
    Error = cudaSetDevice(tid);
    if (DEBUG) printf("CUDA error (cudaSetDevice) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_u,local);
    if (DEBUG) printf("CUDA error (cudaMalloc d_u) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_un,local);
    if (DEBUG) printf("CUDA error (cudaMalloc d_un) in thread %d = %s\n",tid,cudaGetErrorString(Error));
   }
  if (phase==2) {
    // Free local domain in on host
    //free(*h_ul);
    // Free local domain variable on device
    Error = cudaFree(*d_u);
    if (DEBUG) printf("CUDA error (cudaFree d_u) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaFree(*d_un);
    if (DEBUG) printf("CUDA error (cudaFree d_un) in thread %d = %s\n",tid,cudaGetErrorString(Error));
  }
  if (phase==3) {
    // Free the domain on host
    free(*h_u);
  }
}

void Manage_Comms(int phase, int tid, float **h_u, float **d_u){
  cudaError_t Error;
  if (phase==1) {
    // Copy left and right cells from local domain to global domain
    // h_u[ 1 +tid*SNX] = t_u[ 1 ];
    // h_u[SNX+tid*SNX] = t_u[SNX];
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %) :::\n",phase,tid);
    Error=cudaMemcpy(*h_u+1+tid*SNX,*d_u+1,sizeof(float),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
    Error=cudaMemcpy(*h_u+SNX+tid*SNX,*d_u+SNX,sizeof(float),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
  }
  if (phase==2) {
    // Copy left and right cells from global domain to local domain
    // t_u[  0  ] = h_u[  0  +tid*SNX];
    // t_u[SNX+1] = h_u[SNX+1+tid*SNX];
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %) :::\n",phase,tid);
    Error=cudaMemcpy(*d_u,*h_u+tid*SNX,sizeof(float),cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
    Error=cudaMemcpy(*d_u+SNX+1,*h_u+SNX+1+tid*SNX,sizeof(float),cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
  }
  if (phase==3) {
    // Transfer all data from local domains to global domain
    //for (int i = 0; i < SNX; i++) {
    //  h_u[i+1+tid*SNX] = t_u[i+1];
    //}
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %) :::\n",phase,tid);
    Error=cudaMemcpy(*h_u+1+tid*SNX,*d_u+1,SNX*sizeof(float),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
  }
}


void Set_IC(float *u0){
  // Set initial condition in global domain
  // for (int i = 1; i < NX+1; i++) u0[i] = 0.33;
  // Set Dirichlet boundary conditions in global domain
  u0[0]=0.0;  u0[NX+1]=1.0;
}

void Call_Init(float **u0){
  // Load the initial condition
  Set_IC(*u0);
}

__global__ void Set_GPU_IC(int tid, float *ut0){
  // local threads indexes
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // set initial condition only at "interior" nodes
  if (i>0 & i<SNX+1) {
    //switch (tid) { 
    //case 0: ut0[i] = 0.10; break;
    //case 1: ut0[i] = 0.25; break;
    //case 2: ut0[i] = 0.40; break;
    //case 3: ut0[i] = 0.50; break;
    //case 4: ut0[i] = 0.75; break;
    //case 5: ut0[i] = 0.90; break;    
    // }
   ut0[i] = 0.0;
  }
}

void Call_GPU_Init(int tid, float **ut0){
  // Load the initial condition
  int threads = 128;
  int blocks = ((SNX+2) + threads - 1)/threads;
  Set_GPU_IC<<<blocks,threads>>>(tid,*ut0);
  if (DEBUG) printf("CUDA error (Set_GPU_IC) in thread %d = %s\n",tid,cudaGetErrorString(cudaPeekAtLastError()));
}

__global__ void Laplace1d(float *u, float *un){
  // local threads indexes
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int o =   i  ; // node( j,i ) 
  int r = (i+1); // node(j-1,i)  l--o--r
  int l = (i-1); // node(j,i-1) 

  // only update "interior" nodes
  if(i>0 && i<SNX+1) {
    un[o] = u[o] + KX*(u[r]-2*u[o]+u[l]);
  } else {
    un[o] = u[o];
  }
}

void Call_Laplace(int tid, float **u, float **un){
  // Produce one iteration of the laplace operator
  int threads = 128;
  int blocks = ((SNX+2) + threads - 1)/threads;
  Laplace1d<<<blocks,threads>>>(*u,*un);
  if (DEBUG) printf("CUDA error (Call_Laplace) in thread %d = %s\n",tid,cudaGetErrorString(cudaPeekAtLastError()));
}

void Update_Domain(int tid, float *h_u, float *t_u){
  // Explicitly copy data arrays
  if (DEBUG) printf("Copying thread data into the whole domain (thread %d)\n",tid); 
  for (int i = 0; i < SNX; i++) {
    h_u[i+1+tid*SNX] = t_u[i+1];
  }
}

void Call_Update(int tid, float **h_u, float **t_u){
  // produce explicitly: h_u = t_u
  Update_Domain(tid,*h_u,*t_u);
}

void Save_Results(float *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int i = 0; i < NX+2; i++) {
      fprintf(pFile, "%d\t %g\n",i,u[i]);
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}
