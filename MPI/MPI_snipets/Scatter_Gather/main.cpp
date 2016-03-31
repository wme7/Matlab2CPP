#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int size, rank;
  unsigned int n=3;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int *globaldata;
  int *globalsolv;
  int *localdata;
    
  if (rank == 0) {
    globaldata = (int *)malloc(n*size * sizeof(int));
    globalsolv = (int *)malloc(n*size * sizeof(int));
    for (int i = 0; i < n*size; i++)
      globaldata[i] = i;
    
    printf("Processor %d has data: ", rank);
    for (int i = 0; i < n*size; i++)
      printf("%d ", globaldata[i]);
    printf("\n");
  }

  localdata = (int *)malloc(n * sizeof(int) );

  MPI_Scatter(globaldata, n, MPI_INT, localdata, n, MPI_INT, 0, MPI_COMM_WORLD);
  
  for (int i = 0; i < n; i++) {
    printf("Processor %d has data %d\n", rank, localdata[i]);
    // doing some computation with the local data 
    localdata[i] = localdata[i]*2;
  }
  
  MPI_Gather(localdata, n, MPI_INT, globalsolv, n, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (rank == 0) {
    printf("Processor %d has data: ", rank);
    for (int i = 0; i < n*size; i++)
      printf("%d ", globalsolv[i]);
    printf("\n");
  }
  
  if (rank == 0)
    free(globaldata);
  
  MPI_Finalize();
  return 0;
}
