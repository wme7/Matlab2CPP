#include <mpi.h>

int main( int argc, char *argv[] )
{
  int rank;
  float *ptr = NULL;
  const size_t elements = 32;
  MPI_Status status;

  MPI_Init( NULL, NULL );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  cudaMalloc( (void**)&ptr, elements * sizeof(float) );

  if( rank == 0 )
    MPI_Send( ptr, elements, MPI_FLOAT, 1, 0, MPI_COMM_WORLD );
  if( rank == 1 )
    MPI_Recv( ptr, elements, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status );

  cudaFree( ptr );
  MPI_Finalize();

  return 0;
}
