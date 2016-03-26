
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
  double *h_u;
  double *h_un;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Manage Domain sizes
  nx = Manage_Domain(0,rank,size); MPI_Barrier(MPI_COMM_WORLD);

  // Allocate Memory
  Manage_Memory(0,rank,size,nx,&h_u,&h_un);

  // Build Initial Condition
  if (rank==0) Set_IC(h_u); MPI_Barrier(MPI_COMM_WORLD);

  // Copy to slave processors
  Manage_Memory(1,rank,size,nx,&h_u,&h_un); MPI_Barrier(MPI_COMM_WORLD);

  // Record the starting time.
  if (rank==0) wtime=MPI_Wtime();

  // Solver
  //for (step = 0; step < NO_STEPS; step+=2) {
  //  if (step%1000==0 && rank==0) printf("Step %d of %d\n",step,(int)NO_STEPS);
       
    // Compute stencil
    // Call_Laplace(&h_u,&h_un); // 1st iter
    // Call_Laplace(&h_un,&h_u); // 2nd iter
  //}

  // Collect solutions into process 0
  Manage_Memory(2,rank,size,nx,&h_u,&h_un); MPI_Barrier(MPI_COMM_WORLD);

  // Record the final time.
  if (rank==0) {
    wtime = MPI_Wtime()-wtime;
    printf ("\n Wall clock elapsed seconds = %f\n\n", wtime );      
  }
  
  // Write Solution
  if (rank==0) Save_Results(h_u);

  // Free Memory
  Manage_Memory(3,rank,size,nx,&h_u,&h_un); MPI_Barrier(MPI_COMM_WORLD);

  // Terminate MPI.
  MPI_Finalize();

  // Terminate.
  if (rank==0) {
    printf ("HEAT_MPI:\n" );
    printf ("  Normal end of execution.\n\n" );
  }

  return 0;
}
