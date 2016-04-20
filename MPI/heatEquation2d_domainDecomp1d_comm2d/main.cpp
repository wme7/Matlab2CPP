
#include "heat2d.h"
#include <time.h>

/****************/
/* Main program */
/****************/

int main ( int argc, char *argv[] ) {

  // Solution arrays
  real *g_u; /* will be allocated in ROOT only */ 
  real *t_u;
  real *t_un;

  // Auxiliary variables
  int rank;
  int npcs;
  int step;
  dmn domain;
  double wtime;
  int nbrs[2];

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &npcs);

  // if number of np != Sx*Sy then terminate. 
  if (npcs != SY){
    if (rank==ROOT) 
      fprintf(stderr,"%s: Needs at least %d processors.\n", argv[0], SY);
    MPI_Finalize();
    return 1;
  }

  // verify subsizes
  if (NY%SY!=0) {
    if (rank==ROOT) 
      fprintf(stderr,"%s: Subdomain sizes are not an integer value.\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  // Build a 2D cartessian communicator
  MPI_Comm Comm2d;
  int ndim=2;
  int dim[2]={SY,1}; // domain decomposition subdomains
  int period[2]={false,false}; // periodic conditions
  int reorder={true}; // allow reorder if necesary
  int coord[2];
  MPI_Cart_create(MPI_COMM_WORLD,ndim,dim,period,reorder,&Comm2d);
  MPI_Comm_rank(Comm2d,&rank); // rank wrt to Comm2d
  MPI_Cart_coords(Comm2d,rank,2,coord); // rank coordinates

  // Map of neighbours ranks
  MPI_Cart_shift(Comm2d,0,1,&nbrs[DOWN],&nbrs[UP]);

  // Manage Domain sizes
  domain = Manage_Domain(rank,npcs,coord,nbrs); 

  // Allocate Memory
  Manage_Memory(0,domain,&g_u,&t_u,&t_un);

  // Root mode: Build Initial Condition and scatter it to all processes
  if (domain.rank==ROOT) Call_IC(2,g_u);
  MPI_Scatter(g_u, domain.size, MPI_CUSTOM_REAL, t_u+NX, domain.size, MPI_CUSTOM_REAL, ROOT, Comm2d);

  // Exchage Halo regions
  Manage_Comms(domain,Comm2d,t_u); 
  
  // ROOT mode: Record the starting time.
  if (rank==ROOT) wtime=MPI_Wtime();

  // Asynchronous MPI Solver
  for (step = 0; step < NO_STEPS; step+=2) {
    // print iteration in ROOT mode
    if (rank==ROOT && step%10000==0) printf("  Step %d of %d\n",step,(int)NO_STEPS);
    
    // Exchange Boundaries and compute stencil
    Call_Laplace(domain,&t_u,&t_un); Manage_Comms(domain,Comm2d,t_un); // 1st iter
    Call_Laplace(domain,&t_un,&t_u); Manage_Comms(domain,Comm2d,t_u ); // 2nd iter
  }
  // ROOT mode: Record the final time.
  if (rank==ROOT) {
    wtime = MPI_Wtime()-wtime; printf ("\n Wall clock elapsed seconds = %f\n\n", wtime );
  }
  
  // Gather solutions to ROOT and write solution in ROOT mode
  MPI_Gather(t_u+NX, domain.size, MPI_CUSTOM_REAL, g_u, domain.size, MPI_CUSTOM_REAL, ROOT, Comm2d);
  /*
  // CAREFUL: uncomment only for debugging!
  if (rank==0) Print(t_u,NX,domain.ny+2*R); MPI_Barrier(Comm2d);
  if (rank==1) Print(t_u,NX,domain.ny+2*R); MPI_Barrier(Comm2d);
  if (rank==2) Print(t_u,NX,domain.ny+2*R); MPI_Barrier(Comm2d);
  if (rank==3) Print(t_u,NX,domain.ny+2*R); MPI_Barrier(Comm2d);
  if (rank==0) Print(g_u,NX,NY);
  */
  // save results to file
  if (rank==ROOT) Save_Results(g_u);

  // Free Memory
  Manage_Memory(1,domain,&g_u,&t_u,&t_un); 

  // Terminate MPI.
  MPI_Finalize();

  // ROOT mode: Terminate.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n" );
    printf ("  Normal end of execution.\n\n" );
  }

  return 0;
}
