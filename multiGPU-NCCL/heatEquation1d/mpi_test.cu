
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

#include "nccl.h"
#include "mpi.h"
#include "test_utilities.h"

#define SIZE 128
#define NITERS 1

#define DEBUG 0 // Display all error messages
#define NX 1024 // number of cells in the x-direction 
#define L 10.0 // domain length
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 0.1 // tEnd, output time
#define DX (L/NX) // dx, cell size
#define DT (1/(2*C*(1/DX/DX))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define NO_STEPS (TEND/DT) // No. of time steps
#define R 1 // radius of halo region
#define ROOT 0 // define root processor
#define PI 3.1415926535897932f

int Manage_Domain(int phase, int rank, int size){
  
  // All process have by definition the same domain size
  int nx = (float)NX/size;
  if (NX%size != 0) {
    printf("Sorry, the domain size is should be %d*np = NX.\n",nx);
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  // Have process 0 print out some information.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n\n" );
    printf ("  C++/MPI version\n" );
    printf ("  Solve the 1D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf ("  Commence Simulation: processor rank %d out of %d processors"
    " working with %d +2 cells\n",rank,size,nx);

  // return the nx value
  return nx;
}

void Manage_Memory(int phase, int rank, int size, int nx, double **g_u, double **h_u, double **h_un){
  if (phase==0) {
    // Allocate global domain on ROOT
    if (rank==ROOT) *g_u=(double*)malloc(NX*sizeof(double)); // only exist in ROOT!
    // Allocate local domains on MPI threats with 2 extra slots for halo regions
    *h_u =(double*)malloc((nx+2)*sizeof(double));
    *h_un=(double*)malloc((nx+2)*sizeof(double));
  }
  if (phase==1) {
    // Free global domain on ROOT
    if (rank==ROOT) free(*g_u);
    // Free the domain on host
    free(*h_u);
    free(*h_un);
  }
}

void Call_IC(const int IC, double *u0){
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

void Save_Results(double *u){
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

void Set_DirichletBC(double *u, const int n, const char letter, const double value){
  switch (letter) {
  case 'L': { u[ 1 ]=value; break;}
  case 'R': { u[ n ]=value; break;}
  }
}

void Set_NeumannBC(double *u, const int n, const char letter){
  switch (letter) {
  case 'L': { u[ 1 ]=u[ 2 ]; break;}
  case 'R': { u[ n ]=u[n-1]; break;}
  }
}

void Manage_Comms(int r, int p, int n, double **u) {
  MPI_Status status;
  // Communicate halo regions and impose BCs!
  //if (r== 0 ) Set_DirichletBC(*u,n,'L',0.0); // impose Dirichlet BC u[ 0 ] = 1.0
  if (r== 0 ) Set_NeumannBC(*u,n,'L'); // impose Neumann BC : adiabatic condition 
  if (r > 0 ) MPI_Send(*u + 1,1,MPI_DOUBLE,r-1,1,MPI_COMM_WORLD);         // send u[ 1 ] to   rank-1
  if (r <p-1) MPI_Recv(*u+n+1,1,MPI_DOUBLE,r+1,1,MPI_COMM_WORLD,&status); // recv u[n+1] from rank+1
  if (r <p-1) MPI_Send(*u+n  ,1,MPI_DOUBLE,r+1,2,MPI_COMM_WORLD);         // send u[ n ] to   rank+1
  if (r > 0 ) MPI_Recv(*u    ,1,MPI_DOUBLE,r-1,2,MPI_COMM_WORLD,&status); // recv u[ 0 ] from rank-1
  if (r==p-1) Set_NeumannBC(*u,n,'R'); // impose Neumann BC : adiabatic condition
  //if (r==p-1) Set_DirichletBC(*u,n,'R',1.0); // impose Dirichlet BC u[n+1] = 0.0
}

void Laplace1d(const int n, const double * __restrict__ u, double * __restrict__ un){
  int i, o, r, l;
  // perform laplace operator
  for (i = 0; i < n; i++) {

     o =   i  ; // node( j,i ) 
     r = (i+1); // node(j-1,i)  l--o--r
     l = (i-1); // node(j,i-1)  
     
     // only update "interior" nodes
     if(i>0 && i<n-1) {
       un[o] = u[o] + KX*(u[r]-2*u[o]+u[l]);
     } else {
       un[o] = u[o];
     }
  } 
}

void Call_Laplace(int n, double **u, double **un){
  // Produce one iteration of the laplace operator
  Laplace1d(n+2*R,*u,*un); // +2 halo cells
}

int main(int argc, char *argv[]) {
  
  // Solution arrays
  double *g_u; /* will be allocated in ROOT only */ 
  double *t_u;
  double *t_un;

  // Auxiliary variables
  int nx;
  int rank;
  int size;
  int ngpu;
  int step;
  double wtime;

  // nccl variables
  ncclUniqueId commId;
  ncclResult_t ret;
  
  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //  parse arguments 
  if (argc < size) {
    if (rank == 0)
      printf("Usage : %s <GPU list per rank>\n", argv[0]);
    exit(1);
  }
  int gpu = atoi(argv[rank+1]);

  // We have to set our device before NCCL init
  CUDACHECK(cudaSetDevice(gpu)); 
  printf("rank %d is associated to gpu %d\n",rank,gpu);
  MPI_Barrier(MPI_COMM_WORLD);

  // NCCL Communicator creation
  ncclComm_t comm;
  NCCLCHECK(ncclGetUniqueId(&commId));
  MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
  ret = ncclCommInitRank(&comm, size, commId, rank);
  if (ret != ncclSuccess) {
    printf("NCCL Init failed (%d) '%s'\n", ret, ncclGetErrorString(ret));
    exit(1);
  }


  // CUDA stream creation
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // Initialize input values
  int *dptr;
  CUDACHECK(cudaMalloc(&dptr, SIZE*2*sizeof(int)));
  int *val = (int*) malloc(SIZE*sizeof(int));
  for (int v=0; v<SIZE; v++) {
    val[v] = rank + 1;
  }
  CUDACHECK(cudaMemcpy(dptr, val, SIZE*sizeof(int), cudaMemcpyHostToDevice));

  // Compute final value
  int ref = size*(size+1)/2;

  // Run allreduce
  int errors = 0;
  for (int i=0; i<NITERS; i++) {
    NCCLCHECK(ncclAllReduce((const void*)dptr, (void*)(dptr+SIZE), SIZE, ncclInt, ncclSum, comm, stream));
  }

  // Check results
  cudaStreamSynchronize(stream);
  CUDACHECK(cudaMemcpy(val, (dptr+SIZE), SIZE*sizeof(int), cudaMemcpyDeviceToHost));
  for (int v=0; v<SIZE; v++) {
    if (val[v] != ref) {
      errors++;
      printf("[%d] Error at %d : got %d instead of %d\n", rank, v, val[v], ref);
    }
  }
  CUDACHECK(cudaFree(dptr));

  MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (errors)
      printf("%d errors. Test FAILED.\n", errors);
    else
      printf("Test PASSED.\n");
  }

  MPI_Finalize();
  ncclCommDestroy(comm);
  return errors ? 1 : 0;
}
