/* 
  Coded by Manuel A. Diaz.
  NHRI, 2016.04.12. 
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// neighbours convention
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

// hallo radius
#define R 1 // for the time been this is a fixed param.

// domain decompostion
#define Sx 2 // size in x
#define Sy 2 // size in y

// root processor
#define ROOT 0


void print(int *data, int nx, int ny) {    
  printf("-- Global Memory --\n");
  for (int i=0; i<ny; i++) {
    for (int j=0; j<nx; j++) {
     printf("%3d ", data[i*nx+j]);
    }
    printf("\n");
  }
}


int main(int argc, char **argv) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // if number of np != Sx*Sy then terminate. 
    if (size != Sx*Sy){
      if (rank==ROOT) 
	fprintf(stderr,"%s: Needs at least %d processors.\n", argv[0], Sx*Sy);
      MPI_Finalize();
      return 1;
    }

    // Testing : 
    // A grid of 1x4 subgrids
    /* 
     +-----+-----+-----+-----+
     |  0  |  1  |  2  |  3  |
     |(0,0)|(0,1)|(0,2)|(0,3)|
     +-----+-----+-----+-----+
     */
    // A grid of 4x1 subgrids
    /* 
     +-----+
     |  0  |
     |(0,0)|
     +-----+
     |  1  |
     |(1,0)|
     +-----+
     |  2  |
     |(2,0)|
     +-----+
     |  3  |
     |(3,0)|
     +-----+
     */
    // A grid of 2x2 subgrids
    /* 
     +-----+-----+
     |  0  |  1  |
     |(0,0)|(0,1)|
     +-----+-----+
     |  2  |  3  |
     |(1,0)|(1,1)|
     +-----+-----+
     */

    MPI_Comm Comm2d;
    int ndim = 2;
    int dim[2] = {Sy,Sx};
    int period[2] = {false,false}; // for periodic boundary conditions
    int reorder = {true};
     
    // Setup and build cartesian grid
    MPI_Cart_create(MPI_COMM_WORLD,ndim,dim,period,reorder,&Comm2d);
    MPI_Comm_rank(Comm2d, &rank);
    
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
    printf("P:%2d has neighbours (u,d,l,r): %2d %2d %2d %2d\n",
	     rank,nbrs[UP],nbrs[DOWN],nbrs[LEFT],nbrs[RIGHT]);

    /* array sizes */
    const int NX =8;
    const int NY =8;
    const int nx =NX/Sx;
    const int ny =NY/Sy;
    
    // subsizes verification
    if (NX%Sx!=0 || NY%Sy!=0) {
      if (rank==ROOT) 
	fprintf(stderr,"%s: Subdomain sizes not an integer value.\n", argv[0]);
      MPI_Finalize();
      return 1;
    }

    // build a MPI data type for a subarray in Root processor
    MPI_Datatype global, myGlobal;
    int bigsizes[2]  = {NY,NX};
    int subsizes[2]  = {ny,nx};
    int starts[2] = {0,0};
    MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &global);
    MPI_Type_create_resized(global, 0, nx*sizeof(int), &myGlobal); // resize extend
    MPI_Type_commit(&myGlobal);
    
    // build a MPI data type for a subarray in workers
    MPI_Datatype myLocal;
    int bigsizes2[2]  = {R+ny+R,R+nx+R};
    int subsizes2[2]  = {ny,nx};
    int starts2[2] = {R,R};
    MPI_Type_create_subarray(2, bigsizes2, subsizes2, starts2, MPI_ORDER_C, MPI_INT, &myLocal);
    MPI_Type_commit(&myLocal); // now we can use this MPI costum data type

    // halo data types
    MPI_Datatype xSlice, ySlice;
    MPI_Type_vector(nx, 1,   1   , MPI_INT, &xSlice);
    MPI_Type_vector(ny, 1, nx+2*R, MPI_INT, &ySlice);
    MPI_Type_commit(&xSlice);
    MPI_Type_commit(&ySlice);

    // Allocate 2d big-array in root processor
    int i, j;
    int *bigarray; // to be allocated only in root
    if (rank==ROOT) {
      bigarray = (int*)malloc(NX*NY*sizeof(int));
      for (i=0; i<NY; i++) {
	for (j=0; j<NX; j++) {
	  bigarray[i*NX+j] = i*NX+j;
	}
      }
      // print the big array
      print(bigarray, NX, NY);
    }

    // Allocte sub-array in every np
    int *subarray; 
    subarray = (int*)malloc((R+nx+R)*(R+ny+R)*sizeof(int));
    for (i=0; i<ny+2*R; i++) {
      for (j=0; j<nx+2*R; j++) {
	subarray[i*(nx+2*R)+j] = 0;
      }
    }
    
    // build sendcounts and displacements in root processor
    int sendcounts[size];
    int displs[size];
    if (rank==ROOT) {
        for (i=0; i<size; i++) sendcounts[i]=1;
        int disp = 0; // displacement counter
        for (i=0; i<Sy; i++) {
            for (j=0; j<Sx; j++) {
                displs[i*Sx+j]=disp;  disp+=1; // x-displacements
            }
            disp += Sx*(ny-1); // y-displacements
        } 
    }
    
    // scatter pieces of the big data array 
    MPI_Scatterv(bigarray, sendcounts, displs, myGlobal, 
		 subarray, 1, myLocal, ROOT, Comm2d);
    
    // Exchange x - slices with top and bottom neighbors 
    MPI_Sendrecv(&(subarray[  ny  *(nx+2*R)+1]), 1, xSlice, nbrs[UP]  , 1, 
		 &(subarray[  0   *(nx+2*R)+1]), 1, xSlice, nbrs[DOWN], 1, 
		 Comm2d, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&(subarray[  1   *(nx+2*R)+1]), 1, xSlice, nbrs[DOWN], 2, 
		 &(subarray[(ny+1)*(nx+2*R)+1]), 1, xSlice, nbrs[UP]  , 2, 
		 Comm2d, MPI_STATUS_IGNORE);
    // Exchange y - slices with left and right neighbors 
    MPI_Sendrecv(&(subarray[1*(nx+2*R)+  nx  ]), 1, ySlice, nbrs[RIGHT],3, 
		 &(subarray[1*(nx+2*R)+   0  ]), 1, ySlice, nbrs[LEFT] ,3, 
		 Comm2d, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&(subarray[1*(nx+2*R)+   1  ]), 1, ySlice, nbrs[LEFT] ,4, 
    		 &(subarray[1*(nx+2*R)+(nx+1)]), 1, ySlice, nbrs[RIGHT],4, 
		 Comm2d, MPI_STATUS_IGNORE);
        
    // every processor prints the subarray
    for (int p=0; p<size; p++) {
        if (rank == p) {
            printf("Local process on rank %d is:\n", rank);
            for (i=0; i<ny+2*R; i++) {
                putchar('|');
                for (j=0; j<nx+2*R; j++) {
		  printf("%3d ", subarray[i*(nx+2*R)+j]);
                }
                printf("|\n");
            }
        }
        MPI_Barrier(Comm2d);
    }

    // gather all pieces into the big data array
    MPI_Gatherv(subarray, 1, myLocal, 
    		bigarray, sendcounts, displs, myGlobal, ROOT, Comm2d);
    
    // print the bigarray and free array in root
    if (rank==ROOT) print(bigarray, NX, NY);
    
    // MPI types
    MPI_Type_free(&xSlice);
    MPI_Type_free(&ySlice);
    MPI_Type_free(&myLocal);
    MPI_Type_free(&myGlobal);
    
    // free arrays in workers
    if (rank==ROOT) free(bigarray);
    free(subarray);

    // finalize MPI
    MPI_Finalize();

    return 0;
}

