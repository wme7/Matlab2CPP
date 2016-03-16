
#include "heat3d.h"

void Manage_Memory(int phase, float **h_u, float **h_un, float **d_u, float **d_un){
  if (phase==0) {
    // Allocate whole domain in host (master thread)
    *h_u = (float*)malloc(NZ*NY*NX*sizeof(float));
    *h_un= (float*)malloc(NZ*NY*NX*sizeof(float));
  }
  if (phase==1) {
    // Allocate whole domain in device (GPU thread)
    cudaError_t Error = cudaSetDevice(0);
    if (DEBUG) printf("CUDA error (cudaSetDevice) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_u ,NZ*NY*NX*sizeof(float));
    if (DEBUG) printf("CUDA error (cudaMalloc) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_un,NZ*NY*NX*sizeof(float));
    if (DEBUG) printf("CUDA error (cudaMalloc) = %s\n",cudaGetErrorString(Error));
  }
  if (phase==2) {
    // Free the whole domain variables (master thread)
    free(*h_u);
    free(*h_un);
    cudaError_t Error;
    Error = cudaFree(*d_u);
    if (DEBUG) printf("CUDA error (cudaFree) = %s\n",cudaGetErrorString(Error));
    Error = cudaFree(*d_un);
    if (DEBUG) printf("CUDA error (cudaFree) = %s\n",cudaGetErrorString(Error));
  }
}

void Manage_Comms(int phase, float **h_u, float **d_u) {
  // Manage CPU-GPU communicastions
  if (DEBUG) printf(":::::::: Performing Comms (phase %d) ::::::::\n",phase);
  
  if (phase == 0) {
    // move h_u (from HOST) to d_u (to GPU)
    cudaError_t Error = cudaMemcpy(*d_u,*h_u,NZ*NY*NX*sizeof(float),cudaMemcpyHostToDevice);
    if (DEBUG) printf("CUDA error (memcpy h -> d ) = %s\n",cudaGetErrorString(Error));
  }
  if (phase == 1) {
    // move d_u (from GPU) to h_u (to HOST)
    cudaError_t Error = cudaMemcpy(*h_u,*d_u,NZ*NY*NX*sizeof(float),cudaMemcpyDeviceToHost);
    if (DEBUG) printf("CUDA error (memcpy d -> h ) = %s\n",cudaGetErrorString(Error));
  }
}

void Save_Results(float *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int k = 0;k < NZ; k++) {
      for (int j = 0; j < NY; j++) {
	for (int i = 0; i < NX; i++) {      
	  fprintf(pFile, "%d\t %d\t %d\t %g\n",k,j,i,u[i+NX*j+NY*NX*k]);
	}
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

void Save_Time(float *t){
  // print result to txt file
  FILE *pFile = fopen("time.txt", "w");
  if (pFile != NULL) {
	fprintf(pFile, "%g\n",t);
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/

void Set_IC(float *u0){
  int i, j, k, o, IC; 

  // select IC
  IC=2;

  switch (IC) {
  case 1: {
    for (k = 0; k < NZ; k++) {
      for (j = 0; j < NY; j++) {
	for (i = 0; i < NX; i++) {
	  // set all domain's cells equal to zero
	  o = i+NX*j+NY*NX*k;  u0[o] = 0.0;
	  // set BCs in the domain 
	  if (k==0)    u0[o] = 1.0; // bottom
	  if (k==NZ-1) u0[o] = 1.0; // top
	}
      }
    }
    break;
  }
  case 2: {
    for (k = 0; k < NZ; k++) {
      for (j = 0; j < NY; j++) {
	for (i = 0; i < NX; i++) {
	  // set all domain's cells equal to zero
	  o = i+NX*j+NY*NX*k;  
	  u0[o] = 1.0*exp(
			  -(DX*(i-NX/2))*(DX*(i-NX/2))/2
			  -(DY*(j-NY/2))*(DY*(j-NY/2))/2
			  -(DZ*(k-NZ/2))*(DZ*(k-NZ/2))/4);
	}
      }
    }
    break;
  }
    // here to add another IC
  } 
}

void Call_Init(float **u0){
  // Load the initial condition
  Set_IC(*u0);
}

/************************************/
/* LAPLACE ITERATION FUNCTION - CPU */
/************************************/

void Laplace2d_CPU(float *u,float *un){
  // Using (i,j) = [i+N*j] indexes
  int i, j, k, o, n, s, e, w, t, b;
  for (j = 0; j < NY; j++) {
    for (i = 0; i < NX; i++) {
      for (k = 0; k < NZ; k++) {
	
        o =  i + NX*j +NY*NX*k; // node( j,i,k )      n  b
	n = i+NX*(j+1)+NY*NX*k; // node(j+1,i,k)      | /
	s = i+NX*(j-1)+NY*NX*k; // node(j-1,i,k)      |/
	e = (i+1)+NX*j+NY*NX*k; // node(j,i+1,k)  w---o---e
	w = (i-1)+NX*j+NY*NX*k; // node(j,i-1,k)     /|
	t = i+NX*j+NY*NX*(k+1); // node(j,i,k+1)    / |
	b = i+NX*j+NY*NX*(k-1); // node(j,i,k-1)   t  s

	// only update "interior" nodes
	if(i>0 && i<NX-1 && j>0 && j<NY-1 && k>0 && k<NZ-1) {
	  un[o] = u[o] + 
	    KX*(u[e]-2*u[o]+u[w]) + 
	    KY*(u[n]-2*u[o]+u[s]) + 
	    KZ*(u[t]-2*u[o]+u[b]) ;
	} else {
	  un[o] = u[o];
	}
      }
    } 
  }
}

/************************************************************/
/* LAPLACE ITERATION FUNCTION - GPU - WITHOUT SHARED MEMORY */
/************************************************************/

__global__ void Laplace2d_GPU1(const float * __restrict__ u, float * __restrict__ un){
  int o, n, s, e, w, t, b;
  // Threads id
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  const int j = threadIdx.y + blockIdx.y*blockDim.y;
  const int k = threadIdx.z + blockIdx.z*blockDim.z;

  o =  i + NX*j +NY*NX*k; // node( j,i,k )      n  b
  n = i+NX*(j+1)+NY*NX*k; // node(j+1,i,k)      | /
  s = i+NX*(j-1)+NY*NX*k; // node(j-1,i,k)      |/
  e = (i+1)+NX*j+NY*NX*k; // node(j,i+1,k)  w---o---e  
  w = (i-1)+NX*j+NY*NX*k; // node(j,i-1,k)     /| 
  t = i+NX*j+NY*NX*(k+1); // node(j,i,k+1)    / |
  b = i+NX*j+NY*NX*(k-1); // node(j,i,k-1)   t  s

  // only update threads within the domain
  if(i>0 && i<NX-1 && j>0 && j<NY-1 && k>0 && k<NZ-1) {
    un[o] = u[o] + 
      KX*(u[e]-2*u[o]+u[w]) + 
      KY*(u[n]-2*u[o]+u[s]) + 
      KZ*(u[t]-2*u[o]+u[b]) ;
  } else {
    un[o] = u[o];
  }
}

void Call_CPU_Laplace(float **h_u, float **h_un) {
  // Produce one iteration of the laplace operator
  if (USE_CPU==1) {
    // CPU kernel
    Laplace2d_CPU(*h_u,*h_un);
    if (DEBUG) printf("CPU run (Laplace CPU) \n");
  }
}

void Call_GPU_Laplace(float **d_u, float **d_un) {
  // Produce one iteration of the laplace operator
  dim3 dimGrid, dimBlock;

  if (USE_GPU==1) {
    // GPU - no shared memory
    // set threads and blocks ( naive approach )
    dimGrid =dim3(DIVIDE_INTO(NX,NI),DIVIDE_INTO(NY,NJ),DIVIDE_INTO(NZ,NK)); 
    dimBlock=dim3(NI,NJ,NK);
    Laplace2d_GPU1<<<dimGrid,dimBlock>>>(*d_u,*d_un);
    if (DEBUG) printf("CUDA error (Laplace GPU %d) %s\n",
		      USE_GPU,cudaGetErrorString(cudaPeekAtLastError()));
  }
  cudaError_t Error = cudaDeviceSynchronize();
  if (DEBUG) printf("CUDA error (Laplace GPU %d Synchronize) %s\n",USE_GPU,cudaGetErrorString(Error));
}


