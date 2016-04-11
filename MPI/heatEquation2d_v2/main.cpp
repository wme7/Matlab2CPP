
#include "heat2d.h"
#include <time.h>

/****************/
/* Main program */
/****************/

int main ( int argc, char *argv[] ) {

  // Solution arrays
  real *h_u; /* will be allocated in ROOT only */ 
  real *t_u;
  real *t_un;

  // Auxiliary variables
  int rank;
  int npcs;
  int step;
  dmn domain;
  double wtime;
  int nbrs[4];

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npcs);

  // Build a 2D cartessian communicator
  MPI_Comm Comm2d;
  int ndim=2;
  int dim[2]={4,3}; // domain decomposition subdomains
  int period[2]={false,false}; // periodic conditions
  int reorder=true;
  int coord[2];
  MPI_Cart_create(MPI_COMM_WORLD,ndim,dim,period,reorder,&Comm2d);
  MPI_Cart_coords(Comm2d,rank,2,coord); // rank coordinates
  
  // Map the neighbours ranks
  MPI_Cart_shift(Comm2d,0,1,&nbrs[DOWN],&nbrs[UP]);
  MPI_Cart_shift(Comm2d,1,1,&nbrs[LEFT],&nbrs[RIGHT]);

  // Manage Domain sizes
  domain = Manage_Domain(rank,coord,npcs); 

  // Allocate Memory
  Manage_Memory(0,domain,&h_u,&t_u,&t_un);

  // Root mode: Build Initial Condition 
  if (domain.rank==ROOT) Call_IC(2,h_u);

  // Build a MPI data type for a subarray in Root processor
  MPI_Datatype global, myGlobal;
  int nx = domain.nx;
  int ny = domain.ny;
  int bigsizes[2]  = {NY,NX};
  int subsizes[2]  = {ny,nx};
  int starts[2] = {0,0};
  MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_COSTUM_REAL, &global);
  MPI_Type_create_resized(global, 0, (NX/nx)*sizeof(real), &myGlobal); // extend the type 
  MPI_Type_commit(&myGlobal);
    
  // Build a MPI data type for a subarray in workers
  MPI_Datatype myLocal;
  int bigsizes2[2]  = {NY+2*R,NX+2*R};
  int subsizes2[2]  = {ny,nx};
  int starts2[2] = {R,R};
  MPI_Type_create_subarray(2, bigsizes2, subsizes2, starts2, MPI_ORDER_C, MPI_COSTUM_REAL, &myLocal);
  MPI_Type_commit(&myLocal); // now we can use this MPI costum data type

  // Halo data types
  MPI_Datatype xSlice, ySlice;
  MPI_Type_vector(nx, 1,   1   , MPI_COSTUM_REAL, &xSlice);
  MPI_Type_vector(ny, 1, nx+2*R, MPI_COSTUM_REAL, &ySlice);
  MPI_Type_commit(&xSlice);
  MPI_Type_commit(&ySlice);
  
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

  // Scatter global array data
  MPI_Scatterv(h_u, sendcounts, displs, subarrtype, 
	       t_u, 1, mysubarray2, ROOT, Comm2d);

  // Exchage Halo regions
  //Manage_Comms(domain,&t_u); MPI_Barrier(MPI_COMM_WORLD); 
  
  // ROOT mode: Record the starting time.
  if (rank==ROOT) wtime=MPI_Wtime();

  // Asynchronous MPI Solver
  for (step = 0; step < NO_STEPS; step+=2) {
    // print iteration in ROOT mode
    if (rank==ROOT && step%10000==0) printf("  Step %d of %d\n",step,(int)NO_STEPS);
    
    // Exchange Boundaries and compute stencil
    //Call_Laplace(domain,&t_u,&t_un); Manage_Comms(domain,&t_un); // 1st iter
    //Call_Laplace(domain,&t_un,&t_u); Manage_Comms(domain,&t_u ); // 2nd iter
  }
  MPI_Barrier(MPI_COMM_WORLD); //if (rank==0) Print_SubDomain(domain,t_u);

  // ROOT mode: Record the final time.
  if (rank==ROOT) {
    wtime = MPI_Wtime()-wtime; printf ("\n Wall clock elapsed = %f seconds\n\n", wtime );
  }
  
  // Gather solutions to ROOT and write solution in ROOT mode
  MPI_Gatherv(t_u, 1, mysubarray2, 
	      h_u, sendcounts, displs, subarrtype, ROOT, Comm2d);

  // save results to file
  if (rank==ROOT) Save_Results(h_u);

  // Free Memory
  Manage_Memory(1,domain,&h_u,&t_u,&t_un); MPI_Barrier(MPI_COMM_WORLD);

  // Free MPI_types
  MPI_Type_free(&xSlice);
  MPI_Type_free(&ySlice);
  MPI_Type_free(&global);
  MPI_Type_free(&myLocal);
  MPI_Type_free(&myGlobal);

  // Terminate MPI.
  MPI_Finalize();

  // ROOT mode: Terminate.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n" );
    printf ("  Normal end of execution.\n\n" );
  }

  return STATUS_OK;
}
