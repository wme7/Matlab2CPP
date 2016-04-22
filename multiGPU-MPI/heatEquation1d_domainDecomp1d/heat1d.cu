
#include "heat1d.h"

dmn Manage_Domain(int rank, int npcs, int gpu){

  // Set cuda Device
  cudaSetDevice(gpu); 
  
  // allocate sub-domain for a one-dimensional domain decomposition in the X-direction
  dmn domain;
  domain.gpu = gpu;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.nx = NX/SX;
  domain.size = domain.nx;
  domain.rx = rank;
  
  // Have process 0 print out some information.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n\n" );
    printf ("  C++/MPI version\n" );
    printf ("  Solve the 1D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf ("  Commence Simulation: cpu rank %d out of %d cpus with GPU(%d)"
          " working with %d +%d cells\n",rank,npcs,gpu,domain.nx,2*R);

  // return the domain structure
  return domain;
}

void Manage_Memory(int phase, dmn domain, real **h_u, real **d_u, real **d_un){
  if (phase==0) {
    // Allocate global domain on ROOT
    if (domain.rank==ROOT) *h_u=(real*)malloc(NX*sizeof(real)); 
    // Allocate local domains on MPI threats with 2 extra slots for halo regions
    *d_u =(real*)malloc((domain.nx+2*R)*sizeof(real));
    *d_un=(real*)malloc((domain.nx+2*R)*sizeof(real));
  }
  if (phase==1) {
    // Free global domain on ROOT
    if (domain.rank==ROOT) free(*h_u);
    // Free the domain on host
    free(*d_u);
    free(*d_un);
  }
}

__global__ void Laplace1d(const int nx, const real * __restrict__  u, real * __restrict__ un){
  // local threads indexes
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int o =   i  ; // node( j,i ) 
  int r = (i+1); // node(j-1,i)  l--o--r
  int l = (i-1); // node(j,i-1) 

  // only update "interior" nodes
  if(i>0 && i<nx-1) {
    un[o] = u[o] + KX*(u[r]-2*u[o]+u[l]);
  } else {
    un[o] = u[o];
  }
}

void Call_Laplace(dmn domain, real **u, real **un){
  // Produce one iteration of the laplace operator
  int threads = 128;
  int blocks = (domain.nx + threads - 1)/threads;
  Laplace1d<<<blocks,threads>>>(domain.nx+2*R,*u,*un);
  if (DEBUG) printf("CUDA error (Call_Laplace) = %s\n",cudaGetErrorString(cudaPeekAtLastError()));
}
