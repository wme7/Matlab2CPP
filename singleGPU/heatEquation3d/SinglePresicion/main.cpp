
#include "heat3d.h"
#include <time.h>

//  Define a method for checking error in CUDA calls
#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)
inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
#if defined(DISPL)
  if (error != cudaSuccess)
  {
    printf("checkCuda error at %s:%i: %s\n", file, line, cudaGetErrorString(cudaGetLastError()));
    exit(-1);
  }
#endif
  return;
}

int main() {
  // Initialize variables
  float *h_u; h_u = (float*)malloc(sizeof(float)*(NX*NY*NZ));

  // Set Domain Initial Condition and BCs
  Call_IC(h_u);

  // GPU Memory Arrays
  float *d_u;  checkCuda(cudaMalloc((void**)&d_u, sizeof(float)*(NX*NY)));
  float *d_un; checkCuda(cudaMalloc((void**)&d_un,sizeof(float)*(NX*NY)));

  // Copy Initial Condition from host to device
  checkCuda(cudaMemcpy(d_u, h_u,sizeof(float)*(NX*NY),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_un,h_u,sizeof(float)*(NX*NY),cudaMemcpyHostToDevice));

  // GPU kernel launch parameters
  dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 dimGrid (DIVIDE_INTO(NX, BLOCK_SIZE_X), DIVIDE_INTO(NY, BLOCK_SIZE_Y), DIVIDE_INTO(NZ, BLOCK_SIZE_Z)); 

  // Request computer current time
  time_t t = clock();

  // Solver Loop 
  for (int step=0; step < NO_STEPS; step+=2) {
    if (step%10000==0) printf("Step %d of %d\n",step,(int)NO_STEPS);
      // Compute Laplace
      Call_Laplace(dimGrid,dimBlock,d_u,d_un);
      // Call_Laplace_Texture(dimGrid,dimBlock,d_u,d_un);
    }
  if (DEBUG) printf("CUDA error (Jacobi_Method) %s\n",cudaGetErrorString(cudaPeekAtLastError()));

  // Measure and Report computation time
  t = clock()-t; printf("Computing time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // Copy data from device to host
  checkCuda(cudaMemcpy(h_u,d_u,sizeof(float)*(NX*NY*NZ),cudaMemcpyDeviceToHost));

  // uncomment to print solution to terminal
  if (DEBUG) Print2D(h_u);

  // Write solution to file
  Save_Results(h_u); 

  // Free device memory
  checkCuda(cudaFree(d_u));
  checkCuda(cudaFree(d_un));

  // Reset device
  checkCuda(cudaDeviceReset());

  // Free memory on host and device
  free(h_u);

  return 0;
}
