
#include "heat1d.h"
#include <time.h>

/****************/
/* Main program */
/****************/

int main ( int argc, char *argv[] ) {

  // Auxiliary variables
  int rank;
  int size;
  int step;
  dmn domain;
  double wtime;

  // Solution arrays
  real *h_u; /* will be allocated in ROOT only */ 
  real *t_u;
  real *t_un;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // if number of np != SX then terminate. 
  if (size != SX){
    if (rank==ROOT) fprintf(stderr,"%s: Needs at least %d processors.\n", argv[0], SX);
    MPI_Finalize();
    return 1;
  }

  // verify subsizes
  if (NX%SX!=0) {
    if (rank==ROOT) fprintf(stderr,"%s: Subdomain sizes are not an integer value.\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  // Associate each rank with a GPU
  if (argc < size) {
    if (rank==ROOT) printf("Usage : mpirun -np# %s <GPU list per rank>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  // Manage Domain sizes
  domain = Manage_Domain(rank,size,atoi(argv[rank+1])); MPI_Barrier(MPI_COMM_WORLD);

  // Allocate Memory
  Manage_Memory(0,domain,&h_u,&t_u,&t_un);

  // Root mode: Build Initial Condition and scatter it to all processes
  if (rank==ROOT) Call_IC(2,h_u); 
  MPI_Scatter(h_u, domain.nx, MPI_CUSTOM_REAL, t_u+R, domain.nx, MPI_CUSTOM_REAL, ROOT, MPI_COMM_WORLD);

  // Exchange halo data
  //Manage_Comms(domain,&t_u); MPI_Barrier(MPI_COMM_WORLD);

  // ROOT mode: Record the starting time.
  if (rank==ROOT) wtime=MPI_Wtime();

  // Asynchronous MPI Solver
  for (step = 0; step < NO_STEPS; step+=2) {
    // print iteration in ROOT mode
    if (rank==ROOT && step%10000==0) printf("  Step %d of %d\n",step,(int)NO_STEPS);
    
    // Exchange Boundaries and compute stencil
    //Call_Laplace(nx,&t_u,&t_un); Manage_Comms(rank,size,nx,&t_un); // 1st iter
    //Call_Laplace(nx,&t_un,&t_u); Manage_Comms(rank,size,nx,&t_u ); // 2nd iter
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // ROOT mode: Record the final time.
  if (rank==ROOT) {
    wtime = MPI_Wtime()-wtime; printf ("\n Wall clock elapsed seconds = %f\n\n", wtime );
  }
  
  // Gather solutions to ROOT and write solution in ROOT mode
  MPI_Gather(t_u+2, domain.nx, MPI_CUSTOM_REAL, h_u, domain.nx, MPI_CUSTOM_REAL, ROOT, MPI_COMM_WORLD);
  if (rank==ROOT) Save_Results(h_u);

  // Free Memory
  Manage_Memory(1,domain,&h_u,&t_u,&t_un); 
  MPI_Barrier(MPI_COMM_WORLD);

  // Terminate MPI.
  MPI_Finalize();

  // ROOT mode: Terminate.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n" );
    printf ("  Normal end of execution.\n\n" );
  }

  return 0;
}
