
#include "heat3d.h"

/***********************/
/* AUXILIARY FUCNTIONS */
/***********************/
void Print2D(float *u)
{
    // print a single property on terminal
    for (int k = 0; k < NZ; k++) {
      for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
          printf("%8.2f", u[i+NX*j+NX*NY*k]);
        }
        printf("\n");
      }
      printf("\n\n");
    }
}

void Save_Results(float *u){
  // print result to txt file
  float data;
  FILE *pFile = fopen("result.txt", "w");  
  int XY=NX*NY;
  if (pFile != NULL) {
    for (int k = 0; k < NZ; k++) {
      for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
          data = u[i+NX*j+XY*k];
          fprintf(pFile, "%d\t %d\t %d\t %g\n",k,j,i,data);
        }
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/

void Call_IC(float *__restrict u0){
  int i, j, k, o, IC, XY=NX*NY;

  // select IC
  IC=2;

  switch (IC) {
  case 1: {
    for (k = 0; k < NZ; k++) {
      for (j = 0; j < NY; j++) {
      	for (i = 0; i < NX; i++) {
      	  // set all domain's cells equal to zero
      	  o = i+NX*j+XY*k;  u0[o] = 0.0;
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
          o = i+NX*j+XY*k;  
          u0[o] = 1.0*exp(
            -(DX*(i-NX/2))*(DX*(i-NX/2))/1.5
            -(DY*(j-NY/2))*(DY*(j-NY/2))/1.5
            -(DZ*(k-NZ/2))*(DZ*(k-NZ/2))/12);
        }
      }
    }
    break;
  }
    // here to add another IC
  } 
}

/************************************/
/* LAPLACE ITERATION FUNCTION - CPU */
/************************************/

void Laplace3d_CPU(float *u, float *un){
  // Using (i,j) = [i+N*j+M*N*k] indexes
  int i, j, k, o, n, s, e, w, t, b; 
  int XY=NX*NY;

  for (j = 0; j < NY; j++) {
    for (i = 0; i < NX; i++) {
      for (k = 0; k < NZ; k++) {
	
        o = i+ (NX*j) + (XY*k); // node( j,i,k )      n  b
        n = (i==NX-1) ? o:o+NX; // node(j+1,i,k)      | /
        s = (i==0)    ? o:o-NX; // node(j-1,i,k)      |/
        e = (j==NY-1) ? o:o+1;  // node(j,i+1,k)  w---o---e
        w = (j==0)    ? o:o-1;  // node(j,i-1,k)     /|
        t = (k==NZ-1) ? o:o+XY; // node(j,i,k+1)    / |
        b = (k==0)    ? o:o-XY; // node(j,i,k-1)   t  s

        un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]) + KZ*(u[t]-2*u[o]+u[b]);
      }
    } 
  }
}

/************************************/
/* LAPLACE ITERATION FUNCTION - GPU */
/************************************/

__global__ void Laplace3d_GPU(const float * __restrict__ u, float * __restrict__ un){
  int o, n, s, e, w, t, b;  
  int XY=NX*NY;
  // Threads id
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z + blockIdx.z*blockDim.z;

  o = i+ (NX*j) + (XY*k); // node( j,i,k )      n  b
  n = (i==NX-1) ? o:o+NX; // node(j+1,i,k)      | /
  s = (i==0)    ? o:o-NX; // node(j-1,i,k)      |/
  e = (j==NY-1) ? o:o+1;  // node(j,i+1,k)  w---o---e
  w = (j==0)    ? o:o-1;  // node(j,i-1,k)     /|
  t = (k==NZ-1) ? o:o+XY; // node(j,i,k+1)    / |
  b = (k==0)    ? o:o-XY; // node(j,i,k-1)   t  s

  un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]) + KZ*(u[t]-2*u[o]+u[b]);
}

void Call_Laplace(dim3 numBlocks, dim3 threadsPerBlock, float *d_u, float *d_un) {
  // Produce one iteration of the laplace operator

  Laplace3d_GPU<<<numBlocks,threadsPerBlock>>>(d_u,d_un);
  Laplace3d_GPU<<<numBlocks,threadsPerBlock>>>(d_un,d_u);
  if (DEBUG) printf("CUDA error (Laplace GPU %d) %s\n",
    cudaGetErrorString(cudaPeekAtLastError()));

  cudaError_t Error = cudaDeviceSynchronize();
  if (DEBUG) printf("CUDA error (Laplace GPU %d Synchronize) %s\n",
    cudaGetErrorString(Error));
}
