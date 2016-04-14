#include <mpi.h>

int main( int argc, char *argv[]){
  
  // Initialize variables
  int rank;
  float *ptr = NULL;
  const size_t elements = 32;
  MPI_Status status;
  int tag = 0;

  // Initialize MPI
  MPI_Init( NULL, NULL );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // Allocate a single float in device memory
  cudaMalloc( (void**)&ptr, elements * sizeof(float) );

  // Pass element from device 0 to device 1
  if( rank == 0 ) MPI_Send( ptr, elements, MPI_FLOAT, 1, tag, MPI_COMM_WORLD );
  if( rank == 1 ) MPI_Recv( ptr, elements, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status );

  // clear memory in all devices
  cudaFree( ptr );

  // print output message
  printf("TEST PASSED\n");

  // Finalize MPI
  MPI_Finalize();

  return 0;
}
