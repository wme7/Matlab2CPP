
#include "heat2d.h"
#include <time.h>

/****************/
/* Main program */
/****************/

int main ( int argc, char *argv[] ) {

  // Solution arrays
  double *h_u; /* will be allocated in ROOT only */ 
  double *t_u;
  double *t_un;

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
  MPI_Cart_create(MPI_COMM_WORLD,ndim,dim,period,reorder,&Comm2d);

  // obtain current rank coordinates
  int coord[2]; MPI_Cart_coords(Comm2d,rank,2,coord);
  printf("P:%2d My coordinates are %d %d\n",rank,coord[0],coord[1]);
  
  // Map the neighbours ranks
  MPI_Cart_shift(Comm2d,0,1,&nbrs[UP],&nbrs[DOWN]);
  MPI_Cart_shift(Comm2d,1,1,&nbrs[LEFT],&nbrs[RIGHT]);

  // Manage Domain sizes
  domain = Manage_Domain(rank,npcs,dim); 

  // Allocate Memory
  Manage_Memory(0,domain,&h_u,&t_u,&t_un);

  // Root mode: Build Initial Condition 
  if (domain.rank==ROOT) Call_IC(2,h_u);

  // Build 2d subarray data type and scatter IC to all processes
  int sizes[2]    = { NX , NY };           /* global size */
  int subsizes[2] = {domain.nx,domain.ny}; /* local size */
  int starts[2]   = {0,0};                 /* where this one starts */
  MPI_Datatype type, subarrtype;
  MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
  MPI_Type_create_resized(type, 0, gridsize/procgridsize*sizeof(int), &subarrtype);
  MPI_Type_commit(&subarrtype);
  
  //MPI_Scatter(g_u, domain.size, MPI_DOUBLE, t_u+NX, domain.size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

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
  //MPI_Gather(t_u+NX, domain.size, MPI_DOUBLE, g_u, domain.size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
  if (rank==ROOT) Save_Results(h_u);

  // Free Memory
  Manage_Memory(1,domain,&h_u,&t_u,&t_un); MPI_Barrier(MPI_COMM_WORLD);

  // Terminate MPI.
  MPI_Finalize();

  // ROOT mode: Terminate.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n" );
    printf ("  Normal end of execution.\n\n" );
  }

  return STATUS_OK;
}
