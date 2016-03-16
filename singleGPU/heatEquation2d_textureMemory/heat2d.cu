
#include "heat2d.h"

void Manage_Memory(int phase, float **h_u, float **d_u, float **d_un){
  if (phase==0) {
    // Allocate whole domain in host (master thread)
    *h_u = (float*)malloc(NY*NX*sizeof(float));
  }
  if (phase==1) {
    // Allocate whole domain in device (GPU thread)
    cudaError_t Error = cudaSetDevice(0);
    if (DEBUG) printf("CUDA error (cudaSetDevice) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_u ,NY*NX*sizeof(float));
    if (DEBUG) printf("CUDA error (cudaMalloc) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_un,NY*NX*sizeof(float));
    if (DEBUG) printf("CUDA error (cudaMalloc) = %s\n",cudaGetErrorString(Error));
    
    //cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    //cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    //cudaBindTexture2D(NULL, tex_u,     d_u_tex,     desc, NX, NY, NX*sizeof(float));
    //cudaBindTexture2D(NULL, tex_u_old, d_u_tex_old, desc, NX, NY, NX*sizeof(float));
  }
  if (phase==2) {
    // Free the whole domain variables (master thread)
    free(*h_u);
    cudaError_t Error;
    Error = cudaFree(*d_u);
    if (DEBUG) printf("CUDA error (cudaFree) = %s\n",cudaGetErrorString(Error));
    Error = cudaFree(*d_un);
    if (DEBUG) printf("CUDA error (cudaFree) = %s\n",cudaGetErrorString(Error));
    //cudaUnbindTexture()
  }
}

void Manage_Comms(int phase, float **h_u, float **d_u) {
  // Manage CPU-GPU communicastions
  if (DEBUG) printf(":::::::: Performing Comms (phase %d) ::::::::\n",phase);
  
  if (phase == 0) {
    // move h_u (from HOST) to d_u (to GPU)
    cudaError_t Error = cudaMemcpy(*d_u,*h_u,NY*NX*sizeof(float),cudaMemcpyHostToDevice);
    if (DEBUG) printf("CUDA error (memcpy h -> d ) = %s\n",cudaGetErrorString(Error));
  }
  if (phase == 1) {
    // move d_u (from GPU) to h_u (to HOST)
    cudaError_t Error = cudaMemcpy(*h_u,*d_u,NY*NX*sizeof(float),cudaMemcpyDeviceToHost);
    if (DEBUG) printf("CUDA error (memcpy d -> h ) = %s\n",cudaGetErrorString(Error));
  }
}

void Save_Results(float *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int j = 0; j < NY; j++) {
      for (int i = 0; i < NX; i++) {      
	fprintf(pFile, "%d\t %d\t %g\n",j,i,u[i+NX*j]);
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

__global__ void SetIC_onDevice(float *u){
  // threads id 
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int o = i+NX*j; u[o] = 0.0;
  // but ...
  if (i==0)    u[o] = 0.0;
  if (j==0)    u[o] = 0.0;
  if (i==NX-1) u[o] = 1.0;
  if (j==NY-1) u[o] = 1.0;
}

void Call_GPU_Init(float **u0){
  // Load the initial condition
  dim3 threads(16,16);
  dim3 blocks((NX+16+1)/16,(NY+16+1)/16); 
  SetIC_onDevice<<<blocks, threads>>>(*u0);
}

__global__ void Laplace2d(const float * __restrict__ u, float * __restrict__ un){
  int o, n, s, e, w; 
  // Threads id
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  const int j = threadIdx.y + blockIdx.y*blockDim.y;

  o = i + (NX*j);         // node( j,i,k )      n
  n = (i==NX-1) ? o:o+NX; // node(j+1,i,k)      |
  s = (i==0)    ? o:o-NX; // node(j-1,i,k)   w--o--e
  e = (j==NY-1) ? o:o+1;  // node(j,i+1,k)      |
  w = (j==0)    ? o:o-1;  // node(j,i-1,k)      s

  // only update "interior" nodes
  if(i>0 && i<NX-1 && j>0 && j<NY-1) {
    un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]);
  } else {
    un[o] = u[o];
  }
}
/*
__global__ void Laplace2d_texture(float * __restrict__ un, const bool flag) {
  float P, N, S, E, W;
  //Threads id
  const int i = blockIdx.x * blockDim.x + threadIdx.x ;
  const int j = blockIdx.y * blockDim.y + threadIdx.y ;

  if (flag) {
    P = tex2D(tex_u_old, i, j ); // node( i,j )     N
    N = tex2D(tex_u_old, i,j+1); // node(i,j+1)     |
    S = tex2D(tex_u_old, i,j-1); // node(i,j-1)  W--P--E
    E = tex2D(tex_u_old, i+1,j); // node(i+1,j)     |
    W = tex2D(tex_u_old, i-1,j); // node(i-1,j)     S
  } else {
    P = tex2D(tex_u, i, j ); // node( i,j )     N
    N = tex2D(tex_u, i,j+1); // node(i,j+1)     |
    S = tex2D(tex_u, i,j-1); // node(i,j-1)  W--P--E
    E = tex2D(tex_u, i+1,j); // node(i+1,j)     |
    W = tex2D(tex_u, i-1,j); // node(i-1,j)     S
  }

  // --- Only update "interior" (not boundary) node points
  if (i>0 && i<NX-1 && j>0 && j<NY-1) un[i+j*NX] = P + KX*(E-2*P+W) + KY(N-2*P+S);
}
*/
void Call_Laplace(float **d_u, float **d_un) {
  // Produce one iteration of the laplace operator
  dim3 threads(16,16);
  dim3 blocks((NX+16+1)/16,(NX+16+1)/16); 
  Laplace2d<<<blocks,threads>>>(*d_u,*d_un);
  if (DEBUG) printf("CUDA error (Jacobi_Method) %s\n",cudaGetErrorString(cudaPeekAtLastError()));
  cudaError_t Error = cudaDeviceSynchronize();
  if (DEBUG) printf("CUDA error (Jacobi_Method Synchronize) %s\n",cudaGetErrorString(Error));
}
