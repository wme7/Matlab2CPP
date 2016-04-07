#include <stdio.h>
#include <mpi.h>

// neighbours convention
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

/* Run with 4x3 = 12 processes, otherwise MPI will submit an Error */

int main(int argc, char *argv[]) {

  int rank;
  int size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  
  // Build a 4x3 grid of subgrids
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
     |  9  |  10 |  11 |
     |(3,0)|(3,1)|(3,2)|
     +-----+-----+-----+
   */

  MPI_Comm Comm2d;
  int ndim;
  int dim[2];
  int period[2]; // for periodic conditions
  int reorder;
  int coord[2];
  
  // Setup and build cartesian grid
  ndim=2; dim[0]=4; dim[1]=3; period[0]=false; period[1]=false; reorder=true;
  MPI_Cart_create(MPI_COMM_WORLD,ndim,dim,period,reorder,&Comm2d);

  // Every processor prints it rank and coordinates
  MPI_Cart_coords(Comm2d,rank,2,coord);
  printf("P:%2d My coordinates are %d %d\n",rank,coord[0],coord[1]);

  // In Root mode: ask the rank of processor at coordinates (3,1)
  if(rank==0) {
    int id; // the requested processor id
    coord[0]=3;
    coord[1]=1;
    MPI_Cart_rank(Comm2d,coord,&id);
    printf("The processor at coords (%d, %d) has rank %d\n",coord[0],coord[1],id);
  }

  // Every processor build his neighbour map
  int nbrs[4];
  MPI_Cart_shift(Comm2d,0,1,&nbrs[UP],&nbrs[DOWN]);
  MPI_Cart_shift(Comm2d,1,1,&nbrs[LEFT],&nbrs[RIGHT]);
   
  // prints its neighbours
  if (rank==7) {
    printf("P:%2d has neighbours (u,d,l,r): %2d %2d %2d %2d\n",
	   rank,nbrs[UP],nbrs[DOWN],nbrs[LEFT],nbrs[RIGHT]);
  } 
  // if everything looks good up to here, I'll perform a communication test.
  MPI_Barrier(MPI_COMM_WORLD);

  // Making a communication test
  MPI_Request reqSendRecv[8]; // every processor sends 4 INTs and receives 4 INTs
  MPI_Status status[8];

  int out = rank; // communicate the rank number
  int in[4] = {}; // empty array
  int tag = 2; // tag

  for (int i = 0; i < 4; i++) { // following the neighbours order!!
    MPI_Isend( &out ,1,MPI_INT,nbrs[i],tag,MPI_COMM_WORLD,&reqSendRecv[ i ]);
    MPI_Irecv(&in[i],1,MPI_INT,nbrs[i],tag,MPI_COMM_WORLD,&reqSendRecv[i+4]);
  }
  MPI_Waitall(8,reqSendRecv,status);

  // print the communication output
  printf("P:%2d recived from ngbr(u,d,l,r): %2d %2d %2d %2d\n",
	   rank,in[UP],in[DOWN],in[LEFT],in[RIGHT]);

  MPI_Finalize();

  return 0;
}
