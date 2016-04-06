
#include "heat1d.h"
#include <time.h>

/****************/
/* Main program */
/****************/

int main ( int argc, char *argv[] ) {

  // Auxiliary variables
  int nx;
  int rank;
  int size;
  int step;
  double wtime;

  // Solution arrays
  double *g_u; /* will be allocated in ROOT only */ 
  double *t_u;
  double *t_un;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Manage Domain sizes
  nx = Manage_Domain(0,rank,size); 

  // Allocate Memory
  Manage_Memory(0,rank,size,nx,&g_u,&t_u,&t_un);

  // Root mode: Build Initial Condition and scatter it to all processes
  if (rank==ROOT) Call_IC(2,g_u); 
  MPI_Scatter(g_u, nx, MPI_DOUBLE, t_u+R, nx, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  // Exchange halo data
  Manage_Comms(rank,size,nx,&t_u); MPI_Barrier(MPI_COMM_WORLD);

  // ROOT mode: Record the starting time.
  if (rank==ROOT) wtime=MPI_Wtime();

  // Asynchronous MPI Solver
  for (step = 0; step < NO_STEPS; step+=2) {
    // print iteration in ROOT mode
    if (rank==ROOT && step%10000==0) printf("  Step %d of %d\n",step,(int)NO_STEPS);
    
    // Exchange Boundaries and compute stencil
    Call_Laplace(nx,&t_u,&t_un); Manage_Comms(rank,size,nx,&t_un); // 1st iter
    Call_Laplace(nx,&t_un,&t_u); Manage_Comms(rank,size,nx,&t_u ); // 2nd iter
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // ROOT mode: Record the final time.
  if (rank==ROOT) {
    wtime = MPI_Wtime()-wtime;
    printf ("\n Wall clock elapsed seconds = %f\n\n", wtime );
  }
  
  // Gather solutions to ROOT and write solution in ROOT mode
  MPI_Gather(t_u+2, nx, MPI_DOUBLE, g_u, nx, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
  if (rank==ROOT) Save_Results(g_u);

  // Free Memory
  Manage_Memory(1,rank,size,nx,&g_u,&t_u,&t_un); 
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
