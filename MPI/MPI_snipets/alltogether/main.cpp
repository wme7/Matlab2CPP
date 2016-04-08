#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// neighbours convention
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

// hallo radius
#define R 2

void print(int *data, int n) {    
  printf("-- output --\n");
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
     printf("%3d ", data[i*n+j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // if number of np != 9 then terminate. 
    if (size != 9){
        if (rank == 0)
            fprintf(stderr,"%s: Needs at least %d processors.\n", argv[0], 9);
        MPI_Finalize();
        return 1;
    }

    // Build a 3x3 grid of subgrids
    /* 
     +-----+-----+-----+
     |  0  |  1  |  2  |
     |(0,0)|(0,1)|(0,2)|
     +-----+-----+-----+
     |  3  |  4  |  5  |
     |(1,0)|(1,1)|(1,2)|
     +-----+-----+-----+
     |  6  |  7  |  8  |
     |(2,0)|(2,1)|(2,2)|
     +-----+-----+-----+
     */

    MPI_Comm Comm2d;
    int ndim;
    int dim[2];
    int period[2]; // for periodic conditions
    int reorder;
     
    // Setup and build cartesian grid
    ndim=2; dim[0]=3; dim[1]=3; period[0]=false; period[1]=false; reorder=true;
    MPI_Cart_create(MPI_COMM_WORLD,ndim,dim,period,reorder,&Comm2d);
    
    // Every processor prints it rank and coordinates
    int coord[2]; 
    MPI_Cart_coords(Comm2d,rank,2,coord);
    printf("P:%2d My coordinates are %d %d\n",rank,coord[0],coord[1]);
    MPI_Barrier(Comm2d);

    // Every processor build his neighbour map
    int nbrs[4];
    MPI_Cart_shift(Comm2d,0,1,&nbrs[UP],&nbrs[DOWN]);
    MPI_Cart_shift(Comm2d,1,1,&nbrs[LEFT],&nbrs[RIGHT]);
    MPI_Barrier(Comm2d);

    // prints its neighbours
    if (rank==4) {
      printf("P:%2d has neighbours (u,d,l,r): %2d %2d %2d %2d\n",
	     rank,nbrs[UP],nbrs[DOWN],nbrs[LEFT],nbrs[RIGHT]);
    }

    /* array sizes */
    const int bigsize =9;
    const int subsize =3;

// build a MPI data type for a subarray in Root processor
MPI_Datatype mysubarray;
int bigsizes[2]  = {bigsize,bigsize};
int subsizes[2]  = {subsize,subsize};
int starts[2] = {0,0};
MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &mysubarray);
MPI_Type_commit(&mysubarray); // now we can use this MPI costum data type

// build a MPI data type for a subarray in workers
MPI_Datatype mysubarray2;
int bigsizes2[2]  = {subsize+2*R,subsize+2*R};
int subsizes2[2]  = {subsize,subsize};
int starts2[2] = {R,R};
MPI_Type_create_subarray(2, bigsizes2, subsizes2, starts2, MPI_ORDER_C, MPI_INT, &mysubarray2);
MPI_Type_commit(&mysubarray2); // now we can use this MPI costum data type

    // if rank = root processor
    if (rank == 0) {
      // Allocate 2d array
      int *bigarray; 
      bigarray = (int*)malloc(bigsize*bigsize*sizeof(int));
      for (int i=0; i<bigsize; i++)
	for (int j=0; j<bigsize; j++)
	  bigarray[i*bigsize+j] = i*bigsize+j;
      
      // print the big array
      print(bigarray, bigsize);

      // send a 3x3 piece of the big data array 
      MPI_Send(bigarray, 1, mysubarray, 4, 1, Comm2d);

      // we dont need anymore the big array stored in root processor
      MPI_Type_free(&mysubarray);
      free(bigarray);

} else if(rank==4) {
      
      // Allocte sub-array
      int *subarray; 
      subarray = (int*)malloc(bigsize*bigsize*sizeof(int));
      for (int i=0; i<subsize+2*R; i++)
	for (int j=0; j<subsize+2*R; j++)
	  subarray[i*(subsize+2*R)+j] = 0;
      
      // recieve the sub-array
      MPI_Recv(subarray, subsize*subsize, mysubarray2, 0, 1, Comm2d, MPI_STATUS_IGNORE);
	
      // reciver processor prints the subarray
      print(subarray, subsize+2*R);
      
      // we dont need the subarray anymore
      MPI_Type_free(&mysubarray2);
      free(subarray);
    }

    MPI_Finalize();
    return 0;
}

