/* 
  Coded by Manuel A. Diaz.
  NHRI, 2016.04.13. 
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// neighbours convention
#define BOTTOM 0
#define TOP    1
#define NORTH  2
#define SOUTH  3
#define WEST   4
#define EAST   5


// hallo radius
#define R 1 // for the time been this is a fixed param.

// domain decompostion
#define SX 2 // size in x
#define SY 2 // size in y
#define SZ 2 // size in z

// root processor
#define ROOT 0


void printGlobal(int *data, int nx, int ny, int nz) {    
  printf("-- Global Memory --\n");
  for (int k=0; k<nz; k++) {
    printf("-- layer %d --\n",k);
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
	printf("%3d ", data[i+nx*j+nx*ny*k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}


int main(int argc, char **argv) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // if number of np != Sx*Sy then terminate. 
    if (size != SX*SY*SZ){
      if (rank==ROOT) 
	fprintf(stderr,"%s: Needs at least %d processors.\n", argv[0], SX*SY*SZ);
      MPI_Finalize();
      return 1;
    }

    // Testing : 
    // A grid of 2x2x2 subgrids
    /* - layer 0 - TOP
     +-------+-------+
     |   0   |   1   |
     |(0,0,0)|(0,0,1)|
     +-------+-------+
     |   2   |   3   |
     |(0,1,0)|(0,1,1)|
     +-------+-------+
     */
    /* - layer 1 - BOTTOM
     +-------+-------+
     |   4   |   5   |
     |(1,0,0)|(1,0,1)|
     +-------+-------+
     |   6   |   7   |
     |(1,1,0)|(1,1,1)|
     +-------+-------+
     */

    MPI_Comm Comm3d;
    int ndim = 3;
    int dim[3] = {SZ,SY,SX};
    int period[3] = {false,false,false}; // for periodic boundary conditions
    int reorder = {true};
     
    // Setup and build cartesian grid
    MPI_Cart_create(MPI_COMM_WORLD,ndim,dim,period,reorder,&Comm3d);
    MPI_Comm_rank(Comm3d, &rank);
    
    // Every processor prints it rank and coordinates
    int coord[3]; 
    MPI_Cart_coords(Comm3d,rank,3,coord);
    printf("P:%2d My coordinates are %2d %2d %2d\n",rank,coord[0],coord[1],coord[2]);
    MPI_Barrier(Comm3d);

    // Every processor build his neighbour map
    int nbrs[6];
    MPI_Cart_shift(Comm3d,0,1,&nbrs[TOP],&nbrs[BOTTOM]);
    MPI_Cart_shift(Comm3d,1,1,&nbrs[NORTH],&nbrs[SOUTH]);
    MPI_Cart_shift(Comm3d,2,1,&nbrs[WEST],&nbrs[EAST]);
    MPI_Barrier(Comm3d);

    // prints its neighbours
    printf("P:%2d has neighbours (t,b,n,s,w,e): %2d %2d %2d %2d %2d %2d\n",
	   rank,nbrs[TOP],nbrs[BOTTOM],nbrs[NORTH],nbrs[SOUTH],nbrs[WEST],nbrs[EAST]);

    /* array sizes */
    const int NX =6;
    const int NY =4;
    const int NZ =2;
    const int nx =NX/SX;
    const int ny =NY/SY;
    const int nz =NZ/SZ;

    // subsizes verification
    if (NX%SX!=0 || NY%SY!=0 || NZ%SZ!=0) {
      if (rank==ROOT) 
	fprintf(stderr,"%s: Subdomain sizes not an integer value.\n", argv[0]);
      MPI_Finalize();
      return 1;
    }

    // build a MPI data type for a subarray in Root processor
    MPI_Datatype global, myGlobal;
    int bigsizes[3]  = {NZ,NY,NX};
    int subsizes[3]  = {nz,ny,nx};
    int starts[3] = {0,0,0};
    MPI_Type_create_subarray(3, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &global);
    MPI_Type_create_resized(global, 0, nx*sizeof(int), &myGlobal); // resize extend
    MPI_Type_commit(&myGlobal);
    
    // build a MPI data type for a subarray in workers
    MPI_Datatype myLocal;
    int bigsizes2[3]  = {R+nz+R,R+ny+R,R+nx+R};
    int subsizes2[3]  = {nz,ny,nx};
    int starts2[3] = {R,R,R};
    MPI_Type_create_subarray(3, bigsizes2, subsizes2, starts2, MPI_ORDER_C, MPI_INT, &myLocal);
    MPI_Type_commit(&myLocal); // now we can use this MPI costum data type

    // halo data types
    /*MPI_Datatype xSlice, ySlice;
    MPI_Type_vector(nx, 1,   1   , MPI_INT, &xSlice);
    MPI_Type_vector(ny, 1, nx+2*R, MPI_INT, &ySlice);
    MPI_Type_commit(&xSlice);
    MPI_Type_commit(&ySlice);*/

    // Allocate 2d big-array in root processor
    int i, j, k;
    int *bigarray; // to be allocated only in root
    if (rank==ROOT) {
      bigarray = (int*)malloc(NX*NY*NZ*sizeof(int));
      for (k=0; k<NZ; k++) {
	for (j=0; j<NY; j++) {
	  for (i=0; i<NX; i++) {
	    bigarray[i+NX*j+NX*NY*k] = i+NX*j+NX*NY*k;
	  }
	}
      }
      // print the big array
	printGlobal(bigarray,NX,NY,NZ);
    }

    // Allocte sub-array in every np
    int *subarray; 
    subarray = (int*)malloc((R+nx+R)*(R+ny+R)*(R+nz+R)*sizeof(int));
    for (k=0; k<nz+2*R; k++) {
      for (j=0; j<ny+2*R; j++) {
	for (i=0; i<nx+2*R; i++) {
	  subarray[i+(R+nx+R)*j+(R+nx+R)*(R+ny+R)*k] = 0;
	}
      }
    }
    
    // build sendcounts and displacements in root processor
    int sendcounts[size], displs[size];
    if (rank==ROOT) {
        for (i=0; i<size; i++) sendcounts[i]=1;
        int disp = 0; // displacement counter
	for (k=0; k<SZ; k++) {
	  for (j=0; j<SY; j++) {
            for (i=0; i<SX; i++) {
                displs[i+SX*j+SX*SY*k]=disp;  disp+=1; // x-displacements
            }
            disp += SX*(ny-1); // y-displacements
	  }
	  disp += SX*(ny-1)*(nz-1); // z-displacements
        } 
    }
    
    // scatter pieces of the big data array 
    MPI_Scatterv(bigarray, sendcounts, displs, myGlobal, 
		 subarray, 1, myLocal, ROOT, Comm3d);
		 /*
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
		 Comm2d, MPI_STATUS_IGNORE);*/
        
    // every processor prints the subarray
    for (int p=0; p<size; p++) {
      if (rank == p) {
	printf("Local process on rank %d is:\n", rank);
	for (k=0; k<nz+2*R; k++) {
	  printf("-- layer %d --\n",k);
	  for (j=0; j<ny+2*R; j++) {
	    putchar('|');
	    for (i=0; i<nx+2*R; i++) printf("%3d ",subarray[i+(nx+2*R)*j+(nx+2*R)*(ny+2*R)*k]);
	    printf("|\n");
	  }
	  printf("\n");
	}
      }
      MPI_Barrier(Comm3d);
    }

    // gather all pieces into the big data array
    MPI_Gatherv(subarray, 1, myLocal, 
    		bigarray, sendcounts, displs, myGlobal, ROOT, Comm3d);
    
    // print the bigarray and free array in root
    if (rank==ROOT) printGlobal(bigarray,NX,NY,NZ);
    
    // MPI types
    //MPI_Type_free(&xSlice);
    //MPI_Type_free(&ySlice);
    MPI_Type_free(&myLocal);
    MPI_Type_free(&myGlobal);
    
    // free arrays in workers
    if (rank==ROOT) free(bigarray);
    free(subarray);

    // finalize MPI
    MPI_Finalize();

    return 0;
}

