
/****************************************************************/
/* 2D Heat Equation solver with and MPI-2D domain decomposition */
/****************************************************************/

/* 
   Coded by Manuel A. Diaz 
   NHRI, 2016.04.12
 */

#include "heat2d.h"
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
  int nbrs[4];
  int i, j;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // if number of np != Sx*Sy then terminate. 
  if (size != SX*SY){
    if (rank==ROOT) 
      fprintf(stderr,"%s: Needs at least %d processors.\n", argv[0], SX*SY);
    MPI_Finalize();
    return 1;
  }

  // verify subsizes
  if (NX%SX!=0 || NY%SY!=0) {
    if (rank==ROOT) 
      fprintf(stderr,"%s: Subdomain sizes not an integer value.\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  // Build a 2D cartessian communicator
  MPI_Comm Comm2d;
  int ndim=2;
  int dim[2]={SX,SY}; // domain decomposition subdomains
  int period[2]={false,false}; // periodic conditions
  int reorder={true}; // allow reorder if necesary
  int coord[2];
  MPI_Cart_create(MPI_COMM_WORLD,ndim,dim,period,reorder,&Comm2d);
  MPI_Comm_rank(Comm2d,&rank); // rank wrt to Comm2d
  MPI_Cart_coords(Comm2d,rank,2,coord); // rank coordinates
  
  // Map the neighbours ranks
  MPI_Cart_shift(Comm2d,0,1,&nbrs[DOWN],&nbrs[UP]);
  MPI_Cart_shift(Comm2d,1,1,&nbrs[LEFT],&nbrs[RIGHT]);

  // Manage Domain sizes
  domain = Manage_Domain(rank,size,coord,nbrs); 

  // Allocate Memory
  Manage_Memory(0,domain,&h_u,&t_u,&t_un);

  // Root mode: Build Initial Condition 
  if (domain.rank==ROOT) Call_IC(2,h_u);

  // Build MPI data types
  MPI_Datatype myGlobal;
  MPI_Datatype myLocal;
  MPI_Datatype xSlice;
  MPI_Datatype ySlice;
  //Manage_DataTypes(0,domain,&xSlice,&ySlice,&myLocal,&myGlobal);

  // Build a MPI data type for a subarray in Root processor
  MPI_Datatype global;
  int nx = domain.nx;
  int ny = domain.ny;
  int bigsizes[2] = {NY,NX};
  int subsizes[2] = {ny,nx};
  int starts[2] = {0,0};
  MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_CUSTOM_REAL, &global);
  MPI_Type_create_resized(global, 0, nx*sizeof(real), &myGlobal); // extend the type 
  MPI_Type_commit(&myGlobal);
    
  // Build a MPI data type for a subarray in workers
  int bigsizes2[2] = {R+ny+R,R+nx+R};
  int subsizes2[2] = {ny,nx};
  int starts2[2] = {R,R};
  MPI_Type_create_subarray(2, bigsizes2, subsizes2, starts2, MPI_ORDER_C, MPI_CUSTOM_REAL, &myLocal);
  MPI_Type_commit(&myLocal); // now we can use this MPI costum data type

  // Halo data types
  MPI_Type_vector(nx, 1,    1  , MPI_CUSTOM_REAL, &xSlice);
  MPI_Type_vector(ny, 1, R+nx+R, MPI_CUSTOM_REAL, &ySlice);
  MPI_Type_commit(&xSlice);
  MPI_Type_commit(&ySlice);
  
  // build sendcounts and displacements in root processor
  int sendcounts[size];
  int displs[size];
  if (rank==ROOT) {
    for (i=0; i<size; i++) sendcounts[i]=1;
    int disp = 0; // displacement counter
    for (j=0; j<SY; j++) {
      for (i=0; i<SX; i++) {
	displs[i+SX*j]=disp;  disp+=1; // x-displacements
      }
      disp += SX*(domain.ny-1); // y-displacements
    } 
  }

  // Scatter global array data
  MPI_Scatterv(h_u, sendcounts, displs, myGlobal, t_u, 1, myLocal, ROOT, Comm2d);
  
  // Exchage Halo regions
  Manage_Comms(domain,Comm2d,xSlice,ySlice,t_u); 
  if (rank==0) Print(t_u,nx+2*R,ny+2*R); MPI_Barrier(Comm2d);
  if (rank==1) Print(t_u,nx+2*R,ny+2*R); MPI_Barrier(Comm2d);
  if (rank==2) Print(t_u,nx+2*R,ny+2*R); MPI_Barrier(Comm2d);
  if (rank==3) Print(t_u,nx+2*R,ny+2*R); MPI_Barrier(Comm2d);
  if (rank==0) Print(h_u,NX,NY);
  
  // ROOT mode: Record the starting time.
  if (rank==ROOT) wtime=MPI_Wtime();

  // Asynchronous MPI Solver
  for (step = 0; step < NO_STEPS; step+=2) {
    // print iteration in ROOT mode
    if (rank==ROOT && step%10000==0) printf("  Step %d of %d\n",step,(int)NO_STEPS);
    
    // Exchange Boundaries and compute stencil
    //Call_Laplace(domain,&t_u,&t_un); Manage_Comms(domain,Comm2d,xSlice,ySlice,t_un); // 1st iter
    //Call_Laplace(domain,&t_un,&t_u); Manage_Comms(domain,Comm2d,xSlice,ySlice,t_u ); // 2nd iter
  }
  
  // ROOT mode: Record the final time.
  if (rank==ROOT) {
    wtime = MPI_Wtime()-wtime; printf ("\n Wall clock elapsed = %f seconds\n\n", wtime );
  }
  
  // gather all pieces into the big data array
  MPI_Gatherv(t_u, 1, myLocal, h_u, sendcounts, displs, myGlobal, ROOT, Comm2d);

  // save results to file
  if (rank==ROOT) Save_Results(h_u);

  // Free MPI types
  //Manage_DataTypes(1,domain,&xSlice,&ySlice,&myLocal,&myGlobal);
  MPI_Type_free(&xSlice);
  MPI_Type_free(&ySlice);
  MPI_Type_free(&myLocal);
  MPI_Type_free(&myGlobal);
  
  
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
