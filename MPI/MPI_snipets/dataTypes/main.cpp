#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void print(int **data, int n);
int **allocarray(int n);

int main(int argc, char **argv) {

    /* array sizes */
    const int bigsize =10;
    const int subsize =5;

    /* communications parameters */
    const int sender  =0; // root processor
    const int receiver=1; // we need at least 1 target receiver!
    const int ourtag  =2;

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // if number of np < 2 then terminate. 
    if (size < receiver+1) {
        if (rank == 0)
            fprintf(stderr,"%s: Needs at least %d  processors.\n", argv[0], receiver+1);
        MPI_Finalize();
        return 1;
    }

    // if rank = root processor
    if (rank == sender) {
      // Allocate 2d array
      int **bigarray = allocarray(bigsize);
      for (int i=0; i<bigsize; i++)
	for (int j=0; j<bigsize; j++)
	  bigarray[i][j] = i*bigsize+j;
      
      // print the big array
      print(bigarray, bigsize);
	
      // build a MPI data type for a subarray in Root processor
      MPI_Datatype mysubarray;
      int starts[2] = {5,3};
      int subsizes[2]  = {subsize,subsize};
      int bigsizes[2]  = {bigsize,bigsize};
      MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &mysubarray);
      MPI_Type_commit(&mysubarray); // now we can use this MPI costum data type

      // send a 5x5 piece of the big data array 
      MPI_Send(&(bigarray[0][0]), 1, mysubarray, receiver, ourtag, MPI_COMM_WORLD);

      // we dont need anymore the big array stored in root processor
      MPI_Type_free(&mysubarray);
      free(bigarray[0]);
      free(bigarray);

    } else if (rank == receiver) {
      
      // Allocte sub-array
      int **subarray = allocarray(subsize);
      for (int i=0; i<subsize; i++)
	for (int j=0; j<subsize; j++)
	  subarray[i][j] = 0;
      
      // recieve the sub-array
      MPI_Recv(&(subarray[0][0]), subsize*subsize, MPI_INT, sender, ourtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
      // reciver processor prints the subarray
      print(subarray, subsize);
      
      // we dont need the subarray anymore
      free(subarray[0]);
      free(subarray);
    }

    MPI_Finalize();
    return 0;
}

void print(int **data, int n) {    
  printf("-- output --\n");
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      printf("%3d ", data[i][j]);
    }
    printf("\n");
  }
}

int **allocarray(int n) {
  int *data = (int*)malloc(n*n*sizeof(int));
  int **arr = (int**)malloc(n*sizeof(int *));

  for (int i=0; i<n; i++) arr[i] = &(data[i*n]);

  return arr;
}
