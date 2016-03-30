
#include "heat2d.h"
#include <time.h>

/****************/
/* Main program */
/****************/

int main ( int argc, char *argv[] ) {

  // Auxiliary variables
  int rank;
  int npcs;
  int step;
  dmn domain;
  double wtime;

  // Solution arrays
  double *g_u; /* will be allocated in ROOT only */ 
  double *t_u;
  double *t_un;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npcs);

  // Manage Domain sizes
  domain = Manage_Domain(rank,npcs); 

  // Allocate Memory
  Manage_Memory(0,domain,&g_u,&t_u,&t_un);

  // Root mode: Build Initial Condition and scatter it to the rest of processors
  if (domain.rank==ROOT) Call_IC(2,g_u);
  MPI_Scatter(g_u, domain.size, MPI_DOUBLE, t_u, domain.size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  // Exchage Halo regions
  //Manage_Comms(rank,size,nx,&t_u); 

  // ROOT mode: Record the starting time.
  if (rank==ROOT) wtime=MPI_Wtime();

  // Asynchronous MPI Solver
  for (step = 0; step < NO_STEPS; step+=2) {
    // print iteration in ROOT mode
    if (rank==ROOT && step%10000==0) printf("  Step %d of %d\n",step,(int)NO_STEPS);
    
    // Exchange Boundaries and compute stencil
    //Call_Laplace(domain,&t_u,&t_un); Manage_Comms(rank,size,nx,&t_un); // 1st iter
    //Call_Laplace(domain,&t_un,&t_u); Manage_Comms(rank,size,nx,&t_u ); // 2nd iter
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // ROOT mode: Record the final time.
  if (rank==ROOT) {
    wtime = MPI_Wtime()-wtime;
    printf ("\n Wall clock elapsed seconds = %f\n\n", wtime );
  }
  
  // Gather solutions to ROOT and write solution in ROOT mode
  MPI_Gather(t_u, domain.size, MPI_DOUBLE, g_u, domain.size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
  if (rank==ROOT) Save_Results(g_u);

  // Free Memory
  Manage_Memory(1,domain,&g_u,&t_u,&t_un); MPI_Barrier(MPI_COMM_WORLD);

  // Terminate MPI.
  MPI_Finalize();

  // ROOT mode: Terminate.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n" );
    printf ("  Normal end of execution.\n\n" );
  }

  return 0;
}
