#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE 4

int main(int argc, char *argv[]) {

  int rank, size;
  int *sendcounts; 
  int *displs;

  int rem = 0;
  int sum = 0;
  char rec_buf[100];

  const int root =0;

  // the data to be distributed
  char data[SIZE][SIZE] = {
    {'a', 'b', 'c', 'd'},
    {'e', 'f', 'g', 'h'},
    {'i', 'j', 'k', 'l'},
    {'m', 'n', 'o', 'p'}
  };
  char solv[SIZE*SIZE] = {};

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  // display data and solve
  if (rank==0) {
    printf("%s \n ", data);
    printf("%s \n ", solv);
  }
  
  // compute the remainder
  rem = (SIZE*SIZE)%size;

  sendcounts = (int*)malloc(size*sizeof(int));
  displs = (int*)malloc(size*sizeof(int));

  // calculate send counts and displacements
  for (int i = 0; i < size; i++) {
    sendcounts[i] = (SIZE*SIZE)/size;
    if (rem > 0) {
      sendcounts[i]++;
      rem--;
    }
    displs[i] = sum;
    sum += sendcounts[i];
  }

  // print calculated send counts and displacements for each process
  if (rank==root) {
    for (int i=0; i < size; i++) {
      printf("sendcounts[%d] = %d\tdispls[%d] = %d\n", i, sendcounts[i], i, displs[i]);
    }
  }

  // divide the data among processes as described by sendcounts and displs
  MPI_Scatterv(&data, sendcounts, displs, MPI_CHAR,
	       &rec_buf, 100, MPI_CHAR,
	       root, MPI_COMM_WORLD);

  // print what each process received
  printf("%d: ",rank);
  for (int i = 0; i < sendcounts[rank]; i++) {
    printf("%c\t",rec_buf[i]);
  }
  printf("\n");

  // take the data back
  MPI_Gatherv(&rec_buf, sendcounts[rank], MPI_CHAR,
	      &solv, sendcounts, displs, MPI_CHAR,
	      root, MPI_COMM_WORLD);

  // display solve
  if (rank==root) {
      printf("%s\t \n", solv);
  }

  MPI_Finalize();
  
  free(sendcounts);
  free(displs);
  
  return 0;
}


