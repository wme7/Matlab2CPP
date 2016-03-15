
#include "heat2d.h"

void Manage_Memory(int phase, float **h_u, float **h_un, float **d_u, float **d_un){
  if (phase==0) {
    // Allocate whole domain in host (master thread)
    *h_u = (float*)malloc(NY*NX*sizeof(float));
    *h_un= (float*)malloc(NY*NX*sizeof(float));
  }
  if (phase==1) {
    // Allocate whole domain in device (GPU thread)
    cudaError_t Error = cudaSetDevice(0);
    if (DEBUG) printf("CUDA error (cudaSetDevice) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_u ,NY*NX*sizeof(float));
    if (DEBUG) printf("CUDA error (cudaMalloc) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_un,NY*NX*sizeof(float));
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

void Set_IC(float *u0){
  int i, j, o, IC; 

  // select IC
  IC=2;

  switch (IC) {
  case 1: {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
	// set all domain's cells equal to zero
	o = i+NX*j;  u0[o] = 0.0;
	// set BCs in the domain 
	if (j==0)    u0[o] = 0.0; // bottom
	if (i==0)    u0[o] = 0.0; // left
	if (j==NY-1) u0[o] = 1.0; // top
	if (i==NX-1) u0[o] = 1.0; // right
      }
    }
    break;
  }
  case 2: {
    float u_bl = 0.7f;
    float u_br = 1.0f;
    float u_tl = 0.7f;
    float u_tr = 1.0f;

    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
	// set all domain's cells equal to zero
	o = i+NX*j;  u0[o] = 0.0;
	// set BCs in the domain 
	if (j==0)    u0[o] = u_bl + (u_br-u_bl)*i/(NX+1); // bottom
	if (j==NY-1) u0[o] = u_tl + (u_tr-u_tl)*i/(NX+1); // top
	if (i==0)    u0[o] = u_bl + (u_tl-u_bl)*j/(NY+1); // left
	if (i==NX-1) u0[o] = u_br + (u_tr-u_br)*j/(NY+1); // right
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

void Laplace2d_CPU(float *u,float *un){
  // Using (i,j) = [i+N*j] indexes
  int i, j, o, n, s, e, w;
  for (j = 0; j < NY; j++) {
    for (i = 0; i < NX; i++) {

        o =  i + NX*j ; // node( j,i )     n
	n = i+NX*(j+1); // node(j+1,i)     |
	s = i+NX*(j-1); // node(j-1,i)  w--o--e
	e = (i+1)+NX*j; // node(j,i+1)     |
	w = (i-1)+NX*j; // node(j,i-1)     s

	// only update "interior" nodes
	if(i>0 && i<NX-1 && j>0 && j<NY-1) {
	  un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]);
	} else {
	  un[o] = u[o];
	}
    }
  } 
}

__global__ void Laplace2d_GPU1(const float * __restrict__ u, float * __restrict__ un){
  int o, n, s, e, w;
  // Threads id
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  const int j = threadIdx.y + blockIdx.y*blockDim.y;

  o =  i + NX*j ; // node( j,i )     n
  n = i+NX*(j+1); // node(j+1,i)     |
  s = i+NX*(j-1); // node(j-1,i)  w--o--e
  e = (i+1)+NX*j; // node(j,i+1)     |
  w = (i-1)+NX*j; // node(j,i-1)     s

  // only update threads within the domain
  if(i>0 && i<NX-1 && j>0 && j<NY-1) {
    un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]);
  } else {
    un[o] = u[o];
  }
}

__global__ void Laplace2d_GPU2(const float * __restrict__ u, float * __restrict__ un){
  int o, n ,s, e, w;
  
  // Threads id
  const int ti = threadIdx.x; const int i = ti + blockIdx.x*blockDim.x; 
  const int tj = threadIdx.y; const int j = tj + blockIdx.y*blockDim.y; 

  // compute domain index
  o = i+NX*( j ); // node( j,i )     n
  n = i+NX*(j+1); // node(j+1,i)     |
  s = i+NX*(j-1); // node(j-1,i)  w--o--e
  e = (i+1)+NX*j; // node(j,i+1)     |
  w = (i-1)+NX*j; // node(j,i-1)     s
  
  // Allocte an array in shared memory, ut: u_temporary
  __shared__ float ut[NI][NJ];
  
  // read from global memory to shared memory
  // (if thread is not outside domain)
  if (o<NY*NX) {ut[ti][tj] = u[o];}
  __syncthreads();

  // if interior elements, then use shared memory
  if (ti>0 && ti<(NI-1) && tj>0 && tj<(NJ-1)) {
    un[o] = u[o]+ 
	KX*(ut[ti+1][tj]-2*ut[ti][tj]+ut[ti-1][tj])+
	KY*(ut[ti][tj+1]-2*ut[ti][tj]+ut[ti][tj-1]);
  }
  // if halo elements, use use global memory 
  else if(i>0 && i<NX-1 && j>0 && j<NY-1) {
    un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]);
  }
}

__global__ void Laplace2d_GPU3(const float * __restrict__ u,float * __restrict__ un){
  // Allocate an array in shared memory, ut: u_temporary
  __shared__ float ut[NI][NJ];

  // Threads id
  const int ti = threadIdx.x; const int i = ti + blockIdx.x*(NI-2); // blockDim.x = NI-2
  const int tj = threadIdx.y; const int j = tj + blockIdx.y*(NJ-2); // blockDim.y = NJ-2

  // compute domain index
  const int o = i+NX*j; 
  
  // read from global memory to shared memory
  // (if thread is not outside domain)
  if (o<NY*NX) {ut[ti][tj] = u[o];}
  __syncthreads();

  // only update if thread is (1.) within the whole domain
  if(i>0 && i<NX-1 && j>0 && j<NY-1) {
    // and (2.) if is not the boundary of the block
    if ((ti>0) && (ti<NI-1) && (tj>0) && (tj<NJ-1)) {
      // update temperatures
      un[o] = u[o]+ 
	KX*(ut[ti+1][tj]-2*ut[ti][tj]+ut[ti-1][tj])+
	KY*(ut[ti][tj+1]-2*ut[ti][tj]+ut[ti][tj-1]);
    }
  } 
}

__global__ void Laplace2d_GPU4(const float * __restrict__ u,float * __restrict__ un){
  int o, n, s, jp1_sh, j_sh, jm1_sh, tmp_sh, ntot, j_iter; bool within;

  // Allocate an array in shared memory, ut: u_temporary
  __shared__ float ut[NI][3];

  // Threads id
  const int ti = threadIdx.x; const int i =  ti  + blockIdx.x*(NI-2); // blockDim.x = NI-2
  const int tj = threadIdx.y; const int j = tj+1 + blockIdx.y*( NJ ); // blockDim.x = NJ

  // compute domain index
  s = i+NX*(j-1); // node(j-1,i) 
  o = i+NX*( j ); // node( j,i ) <-- current thread
  n = i+NX*(j+1); // node(j+1,i) 

  // Initial shared memory planes
  jp1_sh= 0; j_sh = 1; jm1_sh= 2;

  // total number of cells in domain
  ntot = NX*NY; 

  // read first two planes from global memory to shared memory
  // (if thread is not outside domain)
  if (o < ntot) {ut[ti][jp1_sh] = u[o]; ut[ti][j_sh] = u[s];}
  __syncthreads();

  // Is 0<i<NX-1 and 0<ti<NI-1 ?
  within = i>0 && i<NX-1 && ti>0 && ti<NI-1;

  // Iterate over j-indexes
  for (j_iter=0; j_iter < NJ; j_iter++) {
        
    // read in the next plane
    // (if thread is not outside domain)
    if (n < ntot) ut[ti][jp1_sh] = u[n];
    __syncthreads();

    // compute only if (a) thread is within the domain
    // and (b) thread is not on boundary of a thread block
    if (within && k < NY-1) {
      // update temperature
      un[o] = u[o]+ 
	KX*(ut[ti+1][j_sh]-2*ut[ti][j_sh]+ut[ti-1][j_sh])+ 
	KY*(ut[ti][jp1_sh]-2*ut[ti][j_sh]+ut[ti][jm1_sh]);
    }
    __syncthreads();
    
    // Augment index
    s+=NX; o+=NX; n+=NX; k+=1;
        
    // Swap shared memory planes
    tmp_sh=jm1_sh; jm1_sh=j_sh; j_sh=jp1_sh; jp1_sh=tmp_sh;
  } 
}

/* 
__global__ void Laplace2d_GPU6(const float * __restrict__ u, float * __restrict__ un){
  float o, n, w, s, e;
  // Threads id
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  const int j = threadIdx.y + blockIdx.y*blockDim.y;
  
  if (flag) {
    o = tex2D(texIn,i, j ); // node( j,i )     n
    n = tex2D(texIn,i,j+1); // node(j+1,i)     |
    s = tex2D(texIn,i,j-1); // node(j-1,i)  w--o--e
    e = tex2D(texIn,i+1,j); // node(j,i+1)     |
    w = tex2D(texIn,i-1,j); // node(j,i-1)     s
  } else {
    o = tex2D(texOut,i, j ); // node( j,i )     n
    n = tex2D(texOut,i,j+1); // node(j+1,i)     |
    s = tex2D(texOut,i,j-1); // node(j-1,i)  w--o--e
    e = tex2D(texOut,i+1,j); // node(j,i+1)     |
    w = tex2D(texOut,i-1,j); // node(j,i-1)     s
  }

  // only update threads within the domain
  if(i>0 && i<NX-1 && j>0 && j<NY-1) un[i+NX*j] = o + KX*(e-2*o+w) + KY*(n-2*o+s);
}
*/

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
    dimGrid =dim3(DIVIDE_INTO(NX,NI),DIVIDE_INTO(NY,NJ),1); 
    dimBlock=dim3(NI,NJ,1);
    //dimGrid =dim3(DIVIDE_INTO(NX,32),DIVIDE_INTO(NY,32),1); 
    //dimBlock=dim3(32,32,1);
    Laplace2d_GPU1<<<dimGrid,dimBlock>>>(*d_u,*d_un);
    if (DEBUG) printf("CUDA error (Laplace GPU %d) %s\n",
		      USE_GPU,cudaGetErrorString(cudaPeekAtLastError()));
  }
  if (USE_GPU==2) { 
    // GPU - shared memory
    // set threads and blocks ( halo regions are NOT loaded into shared memory )
    dimGrid =dim3(DIVIDE_INTO(NX,NI),DIVIDE_INTO(NY,NJ),1); 
    dimBlock=dim3(NI,NJ,1);
    Laplace2d_GPU2<<<dimGrid,dimBlock>>>(*d_u,*d_un);
    if (DEBUG) printf("CUDA error (Laplace GPU %d) %s\n",
		      USE_GPU,cudaGetErrorString(cudaPeekAtLastError()));
  }
  if (USE_GPU==3) { 
    // GPU - shared memory
    // set threads and blocks ( halo regions ARE loaded into shared memory )
    // need to compute for n-2 nodes
    // for each N_TILE threads, N_TILE-2 compute
    // number of blocks in each dimension is (NX-2)/(NI-2), rounded upwards
    dimGrid =dim3(DIVIDE_INTO(NX-2,NI-2),DIVIDE_INTO(NY-2,NJ-2),1); 
    dimBlock=dim3(NI,NJ,1);
    Laplace2d_GPU3<<<dimGrid,dimBlock>>>(*d_u,*d_un);
    if (DEBUG) printf("CUDA error (Laplace GPU %d) %s\n",
		      USE_GPU,cudaGetErrorString(cudaPeekAtLastError()));
  }
  if (USE_GPU==4) {
    // GPU - shared memory - iterate upwards through block using a line of threads
    // set threads and blocks
    // need to compute for n-2 nodes
    // for each NI_TILE threads, NI_TILE-2 compute
    // number of blocks in each dimension is (n-2)/(N_TILE-2), rounded upwards
    dimGrid =dim3(DIVIDE_INTO(NX-2,NI-2),DIVIDE_INTO(NY-2,NJ),1); 
    dimBlock=dim3(NI,1,1);
    Laplace2d_GPU4<<<dimGrid,dimBlock>>>(*d_u,*d_un);
    if (DEBUG) printf("CUDA error (Laplace GPU %d) %s\n",
		      USE_GPU,cudaGetErrorString(cudaPeekAtLastError()));
  }
  /*
   if (USE_GPU==6) { 
    // GPU - texture memory
    // set threads and blocks 
    dimGrid =dim3(DIVIDE_INTO(NX,NI),DIVIDE_INTO(NY,NJ),1); 
    dimBlock=dim3(NI,NJ,1);
    Laplace2d_GPU6<<<dimGrid,dimBlock>>>(*d_u,*d_un);
    if (DEBUG) printf("CUDA error (Laplace GPU %d) %s\n",
		      USE_GPU,cudaGetErrorString(cudaPeekAtLastError()));
		      }
  */
  cudaError_t Error = cudaDeviceSynchronize();
  if (DEBUG) printf("CUDA error (Laplace GPU %d Synchronize) %s\n",USE_GPU,cudaGetErrorString(Error));
}


