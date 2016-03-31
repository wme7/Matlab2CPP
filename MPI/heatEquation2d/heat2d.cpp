
#include "heat2d.h"

dmn Manage_Domain(int rank, int npcs){
  // allocate domain and its data
  dmn domain;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.nx = NX/SX;
  domain.ny = NY/SY;
  domain.size = domain.nx*domain.ny;
  
  // All process have by definition the same domain dimensions
  if ((NX*NY)%npcs != 0) {
    printf("Sorry, the domain size should be (%d*np) x (%d*1) = NX x NY cells.\n",
	   domain.nx,domain.ny);
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  // Have process 0 print out some information.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n\n" );
    printf ("  C++/MPI version\n" );
    printf ("  Solve the 2D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf ("  Commence Simulation: procs rank %d out of %d cores"
	  " working with (%d +0) x (%d +%d) cells\n",rank,npcs,domain.nx,domain.ny,2*R);

  return domain;
}

void Manage_Memory(int phase, dmn domain, double **g_u, double **h_u, double **h_un){
  if (phase==0) {
    // Allocate global domain on ROOT
    if (domain.rank==ROOT) *g_u=(double*)malloc(NX*NY*sizeof(double)); // only exist in ROOT!
    // Allocate local domains on MPI threats with 2 extra slots for halo regions
    *h_u =(double*)malloc((domain.nx+0)*(domain.ny+2*R)*sizeof(double));
    *h_un=(double*)malloc((domain.nx+0)*(domain.ny+2*R)*sizeof(double));
  }
  if (phase==1) {
    // Free the domain on host
    if (domain.rank==ROOT) free(*g_u);
    free(*h_u);
    free(*h_un);
  }
}

/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/
void Call_IC(const int IC, double * __restrict u0){
  int i, j, o; 
  switch (IC) {
  case 1: {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
	// set all domain's cells equal to zero
	o = i+NX*j;  u0[o] = 0.0;
	// set BCs in the domain 
	if (j==0)    u0[o] = 0.0; // bottom
	if (i==0)    u0[o] = 0.0; // left
	if (j==NY-1) u0[o] = 1.0; // top
	if (i==NX-1) u0[o] = 1.0; // right
      }
    }
    break;
  }
  case 2: {
    float u_bl = 0.7f;
    float u_br = 1.0f;
    float u_tl = 0.7f;
    float u_tr = 1.0f;

    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
	// set all domain's cells equal to zero
	o = i+NX*j;  u0[o] = 0.0;
	// set BCs in the domain 
	if (j==0)    u0[o] = u_bl + (u_br-u_bl)*i/(NX+1); // bottom
	if (j==NY-1) u0[o] = u_tl + (u_tr-u_tl)*i/(NX+1); // top
	if (i==0)    u0[o] = u_bl + (u_tl-u_bl)*j/(NY+1); // left
	if (i==NX-1) u0[o] = u_br + (u_tr-u_br)*j/(NY+1); // right
      }
    }
    break;
  }
  case 3: {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
	// set all domain's cells equal to zero
	o = i+NX*j;  u0[o] = 0.0;
	// set left wall to 1
	if (i==NX-1) u0[o] = 1.0;
      }
    }
    break;
  }
    // here to add another IC
  }
}

void Save_Results(double *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int j = 0; j < NY; j++) {
      for (int i = 0; i < NX; i++) {      
	fprintf(pFile, "%d\t %d\t %g\n",j,i,u[i+NX*j]);
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

void Print_SubDomain(dmn domain, double *u){
  // print result to terminal
  for (int j = 0; j < domain.ny+2*R; j++) {
    for (int i = 0; i < domain.nx+2*0; i++) {      
      printf("%1.2f ",u[i+domain.nx*j]);
    }
    printf("\n");
  }
}

void Manage_Comms(dmn domain, double **u, double **u_old) {
  MPI_Status status; 
  MPI_Request rqSendUp, rqSendDown, rqRecvUp, rqRecvDown;
  const int r = domain.rank;
  const int p = domain.npcs;
  const int n = domain.nx;
  const int m = domain.ny;
  // Communicate halo regions and impose BCs!
  if (r== 0 ) memcpy(*u+n,*u_old+n,n*sizeof(double)); // impose Dirichlet BC u[row 1]
  if (r > 0 ) MPI_Isend(*u+n      ,n,MPI_DOUBLE,r-1,1,MPI_COMM_WORLD,&rqSendDown); // send u[row 1 ] to   rank-1
  if (r <p-1) MPI_Irecv(*u+n*(m+R),n,MPI_DOUBLE,r+1,1,MPI_COMM_WORLD,&rqRecvUp  ); // recv u[row N ] from rank+1
  if (r <p-1) MPI_Isend(*u+n*m    ,n,MPI_DOUBLE,r+1,2,MPI_COMM_WORLD,&rqSendUp  ); // send u[rowN-1] to   rank+1
  if (r > 0 ) MPI_Irecv(*u        ,n,MPI_DOUBLE,r-1,2,MPI_COMM_WORLD,&rqRecvDown); // recv u[row 0 ] from rank-1
  if (r==p-1) memcpy(*u+n*m,*u_old+n*m,n*sizeof(double)); // impose Dirichlet BC u[row N-1]

  // Wait for process to complete
  if(r <p-1) {
    MPI_Wait(&rqSendDown, &status);
    MPI_Wait(&rqRecvUp,   &status);
  }
  if(r > 0 ) {
    MPI_Wait(&rqRecvDown, &status);
    MPI_Wait(&rqSendUp,   &status);
  }
}

void Laplace2d(const int nx, const int ny, const double * __restrict__ u, double * __restrict__ un){
  // Using (i,j) = [i+N*j] indexes
  int o, n, s, e, w;
  for (int j = 0; j < nx; j++) {
    for (int i = 0; i < ny; i++) {

      o =  i + nx*j ; // node( j,i )     n
      n = i+nx*(j+1); // node(j+1,i)     |
      s = i+nx*(j-1); // node(j-1,i)  w--o--e
      e = (i+1)+nx*j; // node(j,i+1)     |
      w = (i-1)+nx*j; // node(j,i-1)     s
      
      // only update "interior" nodes
      if(i>0 && i<nx-1 && j>0 && j<ny-1) {
	un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]);
      } else {
	un[o] = u[o];
      }
    }
  } 
}

void Call_Laplace(dmn domain, double **u, double **un){
  // Produce one iteration of the laplace operator
  Laplace2d(domain.nx,domain.ny+2*R,*u,*un);
}
