#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// neighbours convention
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

// hallo radius
#define R 1

// root processor
#define ROOT 0


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
        if (rank==ROOT)
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
    MPI_Cart_shift(Comm2d,0,1,&nbrs[DOWN],&nbrs[UP]);
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
    MPI_Datatype subarrtype;
    int bigsizes[2]  = {bigsize,bigsize};
    int subsizes[2]  = {subsize,subsize};
    int starts[2] = {0,0};
    MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &mysubarray);
    //MPI_Type_commit(&mysubarray); // now we can use this MPI costum data type
    MPI_Type_create_resized(mysubarray, 0, bigsize/subsize*sizeof(int), &subarrtype);
    MPI_Type_commit(&subarrtype);
    
    // build a MPI data type for a subarray in workers
    MPI_Datatype mysubarray2;
    int bigsizes2[2]  = {subsize+2*R,subsize+2*R};
    int subsizes2[2]  = {subsize,subsize};
    int starts2[2] = {R,R};
    MPI_Type_create_subarray(2, bigsizes2, subsizes2, starts2, MPI_ORDER_C, MPI_INT, &mysubarray2);
    MPI_Type_commit(&mysubarray2); // now we can use this MPI costum data type

    // halo data types
    MPI_Datatype xSlice, ySlice;
    MPI_Type_vector(subsize, 1,     1      , MPI_INT, &xSlice);
    MPI_Type_vector(subsize, 1, subsize+2*R, MPI_INT, &ySlice);
    MPI_Type_commit(&xSlice);
    MPI_Type_commit(&ySlice);

    // Allocate 2d big-array in root processor
    int i, j;
    int *bigarray; // to be allocated only in root
    if (rank==ROOT) {
      bigarray = (int*)malloc(bigsize*bigsize*sizeof(int));
      for (i=0; i<bigsize; i++) {
	for (j=0; j<bigsize; j++) {
	  bigarray[i*bigsize+j] = i*bigsize+j;
	}
      }
      // print the big array
      print(bigarray, bigsize);
    }

    // Allocte sub-array in every np
    int *subarray; 
    subarray = (int*)malloc(bigsize*bigsize*sizeof(int));
    for (i=0; i<subsize+2*R; i++) {
      for (j=0; j<subsize+2*R; j++) {
	subarray[i*(subsize+2*R)+j] = 0;
      }
    }
    
    // build sendcounts and displacements in root processor
    int sendcounts[subsize*subsize];
    int displs[subsize*subsize];
    if (rank==ROOT) {
        for (i=0; i<subsize*subsize; i++) sendcounts[i] = 1;
        int disp = 0; printf("\n");
        for (i=0; i<subsize; i++) {
            for (j=0; j<subsize; j++) {
                displs[i*subsize+j] = disp;
		printf("%d ",disp);
                disp += 1;
            }
            disp += ((bigsize/subsize)-1)*subsize;
        } printf("\n");
    } 

    // scatter 3x3 pieces of the big data array 
    MPI_Scatterv(bigarray, sendcounts, displs, subarrtype, 
		 subarray, 1, mysubarray2, ROOT, Comm2d);
    
    // Exchange x - slices with top and bottom neighbors 
    MPI_Sendrecv(&(subarray[  subsize  *(subsize+2*R)+1]), 1, xSlice, nbrs[UP]  , 1, 
		 &(subarray[     0     *(subsize+2*R)+1]), 1, xSlice, nbrs[DOWN], 1, 
		 Comm2d, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&(subarray[     1     *(subsize+2*R)+1]), 1, xSlice, nbrs[DOWN], 2, 
		 &(subarray[(subsize+1)*(subsize+2*R)+1]), 1, xSlice, nbrs[UP]  , 2, 
		 Comm2d, MPI_STATUS_IGNORE);
    // Exchange y - slices with left and right neighbors 
    MPI_Sendrecv(&(subarray[1*(subsize+2*R)+  subsize  ]), 1, ySlice, nbrs[RIGHT],3, 
		 &(subarray[1*(subsize+2*R)+     0     ]), 1, ySlice, nbrs[LEFT] ,3, 
		 Comm2d, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&(subarray[1*(subsize+2*R)+     1     ]), 1, ySlice, nbrs[LEFT] ,4, 
    		 &(subarray[1*(subsize+2*R)+(subsize+1)]), 1, ySlice, nbrs[RIGHT],4, 
		 Comm2d, MPI_STATUS_IGNORE);

    // selected reciver processor prints the subarray
    if (rank==4) print(subarray, subsize+2*R);
    MPI_Barrier(Comm2d);

    // gather all 3x3 pieces into the big data array
    MPI_Gatherv(subarray, 1, mysubarray2, 
    		bigarray, sendcounts, displs, subarrtype, ROOT, Comm2d);
    
    // print the bigarray and free array in root
    if (rank==ROOT) {
      print(bigarray, bigsize);
      MPI_Type_free(&mysubarray);
      free(bigarray);
    }

    // free arrays in workers
    MPI_Type_free(&mysubarray2);
    free(subarray);

    // finalize MPI
    MPI_Finalize();
    return 0;
}

