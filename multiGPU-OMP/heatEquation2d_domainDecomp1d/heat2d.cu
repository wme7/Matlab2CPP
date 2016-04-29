
#include "heat2d.h"

void Manage_Memory(int phase, int tid, float **h_u, float **h_ul, float **d_u, float **d_un){
  cudaError_t Error;
  size_t global= ( NX+2)*( NY+2)*sizeof(float);
  size_t local = (SNX+2)*(SNY+2)*sizeof(float);
  if (phase==0) {
    // Allocate domain on host
    *h_u = (float*)malloc(global);
   }
  if (phase==1) {
    // Allocate local domain variable on device
    *h_ul = (float*)malloc(local);
    Error = cudaSetDevice(tid); if (DEBUG) printf("CUDA error (cudaSetDevice) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_u ,local); if (DEBUG) printf("CUDA error (cudaMalloc d_u ) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_un,local); if (DEBUG) printf("CUDA error (cudaMalloc d_un) in thread %d = %s\n",tid,cudaGetErrorString(Error));
   }
  if (phase==2) {
    // Free local domain variable on device
    Error = cudaFree(*d_u ); if (DEBUG) printf("CUDA error (cudaFree d_u ) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    Error = cudaFree(*d_un); if (DEBUG) printf("CUDA error (cudaFree d_un) in thread %d = %s\n",tid,cudaGetErrorString(Error));
    free(*h_ul);
  }
  if (phase==3) {
    // Free the domain on host
    free(*h_u);
  }
}

void Manage_Comms(int phase, int tid, float **h_u, float **h_ul, float **d_u){
  cudaError_t Error;
  if (phase==0) {
    // Transfer all data from local domains to global domain
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %) :::\n",phase,tid);
    Error=cudaMemcpy(*d_u,*h_ul,(SNX+2)*(SNY+2)*sizeof(float),cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
  }
  if (phase==1) {
    // Copy left, right, up and down "interior" boundary  cells from local domain to global domain
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %) :::\n",phase,tid);
    for (int j = 0; j < SNY; j++) {
      Error=cudaMemcpy(*h_u+ 1 +tid*SNX+(NX+2)*(j+1),*d_u+ 1 +(SNX+2)*(j+1),sizeof(float),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
      Error=cudaMemcpy(*h_u+SNX+tid*SNX+(NX+2)*(j+1),*d_u+SNX+(SNX+2)*(j+1),sizeof(float),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
    }
    Error=cudaMemcpy(*h_u+1+tid*SNX+(NX+2)* 1 ,*d_u+1+(SNX+2)* 1 ,SNX*sizeof(float),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
    Error=cudaMemcpy(*h_u+1+tid*SNX+(NX+2)*SNY,*d_u+1+(SNX+2)*SNY,SNX*sizeof(float),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
  }
  if (phase==2) {
    // Copy left, right, up and down boundary cells from global domain to local domain
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %) :::\n",phase,tid);
    for (int j = 0; j < SNY; j++) {
      Error=cudaMemcpy(*d_u+  0  +(SNX+2)*(j+1),*h_u+  0  +tid*SNX+(NX+2)*(j+1),sizeof(float),cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
      Error=cudaMemcpy(*d_u+SNX+1+(SNX+2)*(j+1),*h_u+SNX+1+tid*SNX+(NX+2)*(j+1),sizeof(float),cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
    }
    Error=cudaMemcpy(*d_u+1+(SNX+2)*   0   ,*h_u+1+tid*SNX+(NX+2)*   0   ,SNX*sizeof(float),cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
    Error=cudaMemcpy(*d_u+1+(SNX+2)*(SNY+1),*h_u+1+tid*SNX+(NX+2)*(SNY+1),SNX*sizeof(float),cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
  }
  if (phase==3) {
    // Transfer all data from local domains to global domain
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %) :::\n",phase,tid);
    for (int j = 0; j < SNY; j++) {
      Error=cudaMemcpy(*h_u+1+tid*SNX+(NX+2)*(j+1),*d_u+1+(SNX+2)*(j+1),SNX*sizeof(float),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
    }
  }
  if (phase==4) {
    // Transfer all data from local domains to global domain
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d, thread %) :::\n",phase,tid);
    Error=cudaMemcpy(*h_ul,*d_u,(SNX+2)*(SNY+2)*sizeof(float),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
  }
}


void Set_IC(float *u0){
  // Set Dirichlet boundary conditions in global domain
  for (int i = 0; i < NX+2; i++) u0[   i  +(NX+2)*   0  ]=0.0; // down  
  for (int j = 0; j < NY+2; j++) u0[   0  +(NX+2)*   j  ]=0.0; // left
  for (int i = 0; i < NX+2; i++) u0[   i  +(NX+2)*(NY+1)]=1.0; // up
  for (int j = 0; j < NY+2; j++) u0[(NX+1)+(NX+2)*   j  ]=1.0; // right
}

void Call_CPU_Init(float **u0){
  // Load the initial condition
  Set_IC(*u0);
}

__global__ void Set_GPU_IC(int tid, float *u){
  // Build local threads indexes
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int o = i+(SNX+2)*j; u[o] = 0.0;

  // Set initial condition only at "interior" nodes
  //if (o<(SNX+2)*(SNY+2)) {
  //if (i>0 && i<SNX+1 && j>0 && j<SNY+1) {
      //switch (tid) {
      //case 0: u[o] = 0.10; break;
      //case 1: u[o] = 0.25; break;
      //case 2: u[o] = 0.40; break;
      //case 3: u[o] = 0.50; break;
      //case 4: u[o] = 0.75; break;
      //case 5: u[o] = 0.90; break;
      //}
  //}
  //}
}

void Call_GPU_Init(int tid, float **ut0){
  // Load the initial condition
  dim3 dimBlock(NO_threads,NO_threads); // threads per block
  dim3 dimGrid(ceil((SNX+2.0f)/NO_threads),ceil((SNY+2.0f)/NO_threads)); // blocks in grid
  Set_GPU_IC<<<dimGrid,dimBlock>>>(tid,*ut0);
  if (DEBUG) printf("CUDA error (Set_GPU_IC) in thread %d = %s\n",tid,cudaGetErrorString(cudaPeekAtLastError()));
  cudaError_t Error = cudaDeviceSynchronize();
  if (DEBUG) printf("CUDA error (Set_GPU_IC Synchronize) %s\n",cudaGetErrorString(Error));
}

__global__ void Laplace1d(float *u, float *un){
  // local threads indexes
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  int o = (i + (SNX+2)*j); // node( j,i )     n
  int n = i+(SNX+2)*(j+1); // node(j+1,i)     |
  int s = i+(SNX+2)*(j-1); // node(j-1,i)  w--o--e
  int e = (i+1)+(SNX+2)*j; // node(j,i+1)     |
  int w = (i-1)+(SNX+2)*j; // node(j,i-1)     s

  // only update "interior" nodes
  if(i>0 & i<SNX+1 & j>0 & j<SNY+1) {
    un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]);
  } else {
    un[o] = u[o];
  }
}

void Call_Laplace(int tid, float **u, float **un){
  // Produce one iteration of the laplace operator
  dim3 dimBlock(NO_threads,NO_threads); // threads per block
  dim3 dimGrid(ceil((SNX+2.0f)/NO_threads),ceil((SNY+2.0f)/NO_threads)); // blocks in grid
  Laplace1d<<<dimGrid,dimBlock>>>(*u,*un);
  if (DEBUG) printf("CUDA error (Call_Laplace) in thread %d = %s\n",tid,cudaGetErrorString(cudaPeekAtLastError()));
}

void Save_Results(float *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int j = 0; j < NY+2; j++) {
      for (int i = 0; i < NX+2; i++) {      
	fprintf(pFile, "%d\t %d\t %g\n",i,j,u[i+(NX+2)*j]);
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

void Save_Results_Tid(int tid, float *u){
  // print result to txt file
  if (tid==0) {
    FILE *pFile = fopen("result0.txt", "w");
    if (pFile != NULL) {
      for (int j = 0; j < SNY+2; j++) {
	for (int i = 0; i < SNX+2; i++) {
	  fprintf(pFile, "%d\t %d\t %g\n",i,j,u[i+(SNX+2)*j]);
	}
      }
      fclose(pFile);
    } else {
      printf("Unable to save to file\n");
    }
  }
  if (tid==1) {
    FILE *pFile = fopen("result1.txt", "w");
    if (pFile != NULL) {
      for (int j = 0; j < SNY+2; j++) {
	for (int i = 0; i < SNX+2; i++) {
	  fprintf(pFile, "%d\t %d\t %g\n",i,j,u[i+(SNX+2)*j]);
	}
      }
      fclose(pFile);
    } else {
      printf("Unable to save to file\n");
    }
  }
}
