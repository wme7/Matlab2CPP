
/****************************************************************/
/* 3D Heat Equation solver with and MPI-3D domain decomposition */
/****************************************************************/

/* 
   Coded by Manuel A. Diaz 
   NHRI, 2016.04.12
 */

#include "heat3d.h"
#include <time.h>

int main ( int argc, char *argv[] ) {

  // Solution arrays
  real *h_u; /* to be allocated in ROOT only */ 
  real *t_u;
  real *t_un;

  // Auxiliary variables
  int rank;
  int size;
  int step;
  dmn domain;
  double wtime;
  int nbrs[6];
  int i, j, k;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // if number of np != Sx*Sy*Sz then terminate. 
  if (size != SX*SY*SZ){
    if (rank==ROOT) 
      fprintf(stderr,"%s: Needs at least %d processors.\n", argv[0], SX*SY*SZ);
    MPI_Finalize();
    return 1;
  }

  // verify subsizes
  if (NX%SX!=0 || NY%SY!=0 || NZ%SZ!=0) {
    if (rank==ROOT) 
      fprintf(stderr,"%s: Subdomain sizes not an integer value.\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  // Build a 2D cartessian communicator
  MPI_Comm Comm3d;
  int ndim=3;
  int dim[3]={SZ,SY,SX}; // domain decomposition subdomains
  int period[3]={false,false,false}; // periodic conditions
  int reorder={true}; // allow reorder if necesary
  int coord[3];
  MPI_Cart_create(MPI_COMM_WORLD,ndim,dim,period,reorder,&Comm3d);
  MPI_Comm_rank(Comm3d,&rank); // rank wrt to Comm2d
  MPI_Cart_coords(Comm3d,rank,3,coord); // rank coordinates
  
  // Map the neighbours ranks
  MPI_Cart_shift(Comm3d,0,1,&nbrs[TOP],&nbrs[BOTTOM]);
  MPI_Cart_shift(Comm3d,1,1,&nbrs[NORTH],&nbrs[SOUTH]);
  MPI_Cart_shift(Comm3d,2,1,&nbrs[WEST],&nbrs[EAST]);

  // Manage Domain sizes
  domain = Manage_Domain(rank,size,coord,nbrs); 

  // Allocate Memory
  Manage_Memory(0,domain,&h_u,&t_u,&t_un);

  // Root mode: Build Initial Condition 
  if (domain.rank==ROOT) Call_IC(2,h_u);

  // Build MPI data types
  MPI_Datatype myGlobal;
  MPI_Datatype myLocal;
  MPI_Datatype xySlice;
  MPI_Datatype yzSlice;
  MPI_Datatype xzSlice;
  //Manage_DataTypes(0,domain,&xySlice,&yzSlice,&xzSlice,&myLocal,&myGlobal);

  // Build a MPI data type for a subarray in Root processor
  MPI_Datatype global;
  int nx = domain.nx;
  int ny = domain.ny;
  int nz = domain.nz;
  int bigsizes[3] = {NZ,NY,NX};
  int subsizes[3] = {nz,ny,nx};
  int starts[3] = {0,0,0};
  MPI_Type_create_subarray(3, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_CUSTOM_REAL, &global);
  MPI_Type_create_resized(global, 0, nx*sizeof(real), &myGlobal); // extend the type 
  MPI_Type_commit(&myGlobal);
    
  // Build a MPI data type for a subarray in workers
  int bigsizes2[3] = {R+nz+R,R+ny+R,R+nx+R};
  int subsizes2[3] = {nz,ny,nx};
  int starts2[3] = {R,R,R};
  MPI_Type_create_subarray(3, bigsizes2, subsizes2, starts2, MPI_ORDER_C, MPI_CUSTOM_REAL, &myLocal);
  MPI_Type_commit(&myLocal); // now we can use this MPI costum data type

  // halo data types
  MPI_Datatype yVector;
  MPI_Type_vector( ny, nx, nx+2*R, MPI_CUSTOM_REAL, &xySlice); MPI_Type_commit(&xySlice);
  MPI_Type_vector( ny,  1, nx+2*R, MPI_CUSTOM_REAL, &yVector); 
  MPI_Type_create_hvector(nz, 1, (nx+2*R)*(ny+2*R)*sizeof(real), yVector, &yzSlice); MPI_Type_commit(&yzSlice);
  MPI_Type_vector( nz, nx, (nx+2*R)*(ny+2*R), MPI_CUSTOM_REAL, &xzSlice); MPI_Type_commit(&xzSlice);
  
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
      disp += SX*NY*(nz-1); // z-displacements
    } 
  }

  // Scatter global array data and exchange halo regions
  MPI_Scatterv(h_u, sendcounts, displs, myGlobal, t_u, 1, myLocal, ROOT, Comm3d);
  Manage_Comms(domain,Comm3d,xySlice,yzSlice,xzSlice,t_u); MPI_Barrier(Comm3d);
   
  // ROOT mode: Record the starting time.
  if (rank==ROOT) wtime=MPI_Wtime();

  // Asynchronous MPI Solver
  for (step = 0; step < NO_STEPS; step+=2) {
    // print iteration in ROOT mode
    if (rank==ROOT && step%10000==0) printf("  Step %d of %d\n",step,(int)NO_STEPS);
    
    // Exchange Boundaries and compute stencil
    Call_Laplace(domain,&t_u,&t_un);Manage_Comms(domain,Comm3d,xySlice,yzSlice,xzSlice,t_un);//1stIter
    Call_Laplace(domain,&t_un,&t_u);Manage_Comms(domain,Comm3d,xySlice,yzSlice,xzSlice,t_u );//2ndIter
  }
  
  // ROOT mode: Record the final time.
  if (rank==ROOT) {
    wtime = MPI_Wtime()-wtime; printf ("\n Wall clock elapsed = %f seconds\n\n", wtime );    
  }
  /*
  // CAREFUL: uncomment only for debugging. Print subroutine
  for (int p=0; p<size; p++) {
    if (rank == p) {
      printf("Local process on rank %d is:\n", rank);
      for (k=0; k<nz+2*R; k++) {
	printf("-- layer %d --\n",k);
	for (j=0; j<ny+2*R; j++) {
	  putchar('|');
	  for (i=0; i<nx+2*R; i++) printf("%3.0f ",t_u[i+(nx+2*R)*j+(nx+2*R)*(ny+2*R)*k]);
	  printf("|\n");
	}
	printf("\n");
      }
    }
    MPI_Barrier(Comm3d);
    }*/

  // gather all pieces into the big data array
  MPI_Gatherv(t_u, 1, myLocal, h_u, sendcounts, displs, myGlobal, ROOT, Comm3d);
 
  // save results to file
  //if (rank==0) Print(h_u,NX,NY,NZ);
  if (rank==ROOT) Save_Results(h_u); 

  // Free MPI types
  Manage_DataTypes(1,domain,&xySlice,&yzSlice,&xzSlice,&myLocal,&myGlobal);
  
  // Free Memory
  Manage_Memory(1,domain,&h_u,&t_u,&t_un); 
    
  // finalize MPI
  MPI_Finalize();

  // ROOT mode: Terminate.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n" );
    printf ("  Normal end of execution.\n\n" );
  }

  return 0;
}
