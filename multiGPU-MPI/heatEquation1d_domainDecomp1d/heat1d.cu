
#include "heat1d.h"

void Manage_Devices(){
  // we set devices before MPI init to make sure MPI is aware of the cuda devices!
  char * localRankStr = NULL;
  int rank = 0, devCount = 0;
  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv(ENV_LOCAL_RANK)) != NULL) { rank = atoi(localRankStr); }
  cudaGetDeviceCount(&devCount);
  //cudaSetDevice(rank/devCount);
  cudaSetDevice(rank);
  printf("number of devices found: %d\n",devCount);
}

dmn Manage_Domain(int rank, int npcs, int gpu){
  // allocate sub-domain for a one-dimensional domain decomposition in the X-direction
  dmn domain;
  domain.gpu = gpu;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.nx = NX/SX;
  domain.size = domain.nx;
  domain.rx = rank;
  
  cudaSetDevice(domain.gpu); 

  // Have process 0 print out some information.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n\n" );
    printf ("  C++/MPI version\n" );
    printf ("  Solve the 1D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf ("  Commence Simulation: cpu rank %d out of %d cores with GPU(%d)"
          " working with %d +%d cells\n",rank,npcs,gpu,domain.nx,2*R);

  // return the domain structure
  return domain;
}

void Manage_Memory(int phase, dmn domain, real **h_u, real **t_u, real **d_u, real **d_un){
  size_t global = NX*sizeof(real);
  size_t local = (domain.nx+2*R)*sizeof(real);
  cudaError_t Error;
  if (phase==0) {
    // Allocate global domain on ROOT
    if (domain.rank==ROOT) *h_u=(real*)malloc(global);
    // Allocate local domains on MPI threats with 2 extra slots for halo regions
    *t_u =(real*)malloc(local);
    // Allocate local domains on devices with 2 extra slots for halo regions
    Error = cudaSetDevice(domain.gpu); if (DEBUG) printf("CUDA error (cudaSetDevice) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_u ,local); if (DEBUG) printf("CUDA error (cudaMalloc d_u) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_un,local); if (DEBUG) printf("CUDA error (cudaMalloc d_un) = %s\n",cudaGetErrorString(Error));
  }
  if (phase==1){
    // Free local domain variable on device
    //Error = cudaSetDevice(domain.gpu); if (DEBUG) printf("CUDA error (cudaSetDevice) = %s\n",cudaGetErrorString(Error));
    Error = cudaFree(*d_u ); if (DEBUG) printf("CUDA error (cudaFree d_u) = %s\n",cudaGetErrorString(Error));
    Error = cudaFree(*d_un); if (DEBUG) printf("CUDA error (cudaFree d_un) = %s\n",cudaGetErrorString(Error));
    // Free the local domain on host
    free(*t_u);
  }
  if (phase==2) {
    // Free global domain on ROOT
    if (domain.rank==ROOT) free(*h_u);    
  }
}

/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/
void Call_IC(const int IC, real *u0){
  // Set initial condition in global domain
  switch (IC) {
    case 1: {
      // Uniform Temperature in the domain, temperature will be imposed at boundaries
      for (int i = 0; i < NX; i++) u0[i]=0.0;
      // Set Dirichlet boundary conditions in global domain as u0[0]=0.0;  u0[NX]=1.0; namely
      u0[0]=0.0; u0[NX]=1.0;
      break;
    }
    case 2: {
      // A square jump problem
      for (int i= 0; i < NX; i++) {if (i>0.3*NX && i<0.7*NX) u0[i]=1.0; else u0[i]=0.0;}
      // Set Neumann boundary conditions in global domain u0'[0]=0.0;  u0'[NX]=0.0;
      break;
    }
      // add another IC
  }
}

void Save_Results(real *u){
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

__global__ void Set_DirichletBC(const int n, real *u, const char letter){
  switch (letter) {
  case 'L': { u[ 1 ]=0.0; break;}
  case 'R': { u[ n ]=1.0; break;}
  }
}

__global__ void Set_NeumannBC(const int n, real *u, const char letter){
  switch (letter) {
  case 'L': { u[ 1 ]=u[ 2 ]; break;}
  case 'R': { u[ n ]=u[n-1]; break;}
  }
}

void Manage_Comms(int phase, dmn domain, real **t_u, real **d_u) {
  cudaError_t Error;
  if (phase==0) {
    // Send local domains to their associated GPU
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d) :::\n",phase);
    Error=cudaMemcpy(*d_u,*t_u,(domain.nx+2*R)*sizeof(real),cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
  }
  if (phase==1) {
    // Communicate halo regions 
    int n = domain.nx;
    MPI_Status status;

    // Impose BCs!
    if (domain.rx==  0 ) Set_NeumannBC<<<1,1>>>(domain.nx,*d_u,'L'); // impose Dirichlet BC u[row  1 ]
    if (domain.rx==SX-1) Set_NeumannBC<<<1,1>>>(domain.nx,*d_u,'R'); // impose Dirichlet BC u[row M-1]

    // Communicate halo regions 
    if (domain.rx >  0 ) MPI_Send(*d_u + R,R,MPI_CUSTOM_REAL,domain.rx-1,0,MPI_COMM_WORLD);         // send u[ 1 ] to   rank-1
    if (domain.rx <SX-1) MPI_Recv(*d_u+n+R,R,MPI_CUSTOM_REAL,domain.rx+1,0,MPI_COMM_WORLD,&status); // recv u[n+1] from rank+1
    if (domain.rx <SX-1) MPI_Send(*d_u+n  ,R,MPI_CUSTOM_REAL,domain.rx+1,1,MPI_COMM_WORLD);         // send u[ n ] to   rank+1
    if (domain.rx >  0 ) MPI_Recv(*d_u    ,R,MPI_CUSTOM_REAL,domain.rx-1,1,MPI_COMM_WORLD,&status); // recv u[ 0 ] from rank-1
  }
  if (phase==2) {
    // Collect local domains from their associated GPU
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d) :::\n",phase);
    Error=cudaMemcpy(*t_u,*d_u,(domain.nx+2*R)*sizeof(real),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
  }
}

__global__ void Laplace1d(const int nx, const int rx, const real * __restrict__  u, real * __restrict__ un){
  int o, r, l;
  // Threads id
  const int i = blockDim.x * blockIdx.x + threadIdx.x; 

  o =  i;  // node( j,i ) 
  r = i+1; // node(j-1,i)  l--o--r
  l = i-1; // node(j,i-1) 

  // only update "interior" nodes
  if(i>0 && i<nx-1) {
    un[o] = u[o] + KX*(u[r]-2*u[o]+u[l]);
  } else {
    un[o] = u[o];
  }
}

void Call_Laplace(dmn domain, real **u, real **un){
  // Produce one iteration of the laplace operator
  int threads = 128; // number of threads in x directions
  int blockSize=tds, gridSize=(domain.nx+threads-1)/threads;
  Laplace1d<<<blocks,threads>>>(domain.nx+2*R,domain.rx,*u,*un);
  if (DEBUG) printf("CUDA error in gpu %d (Call_Laplace) = %s\n",domain.gpu,cudaGetErrorString(cudaPeekAtLastError()));
  cudaError_t Error = cudaDeviceSynchronize();
  if (DEBUG) printf("CUDA error (Jacobi_Method Synchronize) %s\n",cudaGetErrorString(Error));
}
