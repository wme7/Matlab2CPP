
#include "heat2d.h"

dmn Manage_Domain(int rank, int npcs){
  // allocate domain and its data
  dmn domain;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.rx = rank%SX;
  domain.ry = rank/SY;
  domain.nx = NX/SX;
  domain.ny = NY/SY;
  domain.size = domain.nx*domain.ny;
  
  // All process have by definition the same domain dimensions
  if ((NX*NY)%npcs != 0) {
    printf("Sorry, the domain size should be (%d*np)*(%d*1) = NX*NY.\n",domain.nx,domain.ny);
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
	  " working with nx=(%d +2) by ny=(%d +2) cells\n",rank,npcs,domain.nx,domain.ny);

  return domain;
}

void Manage_Memory(int phase, dmn domain, double **g_u, double **h_u, double **h_un){
  if (phase==0) {
    // Allocate global domain on ROOT
    if (domain.rank==ROOT) *g_u=(double*)malloc(NX*NY*sizeof(double)); // only exist in ROOT!
    // Allocate local domains on MPI threats with 2 extra slots for halo regions
    *h_u =(double*)malloc((domain.nx+2)*(domain.ny+2)*sizeof(double));
    *h_un=(double*)malloc((domain.nx+2)*(domain.ny+2)*sizeof(double));
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

void Set_DirichletBC(double *u, const int n, const char letter, const double value){
  switch (letter) {
  case 'L': { u[ 1 ]=value; break;}
  case 'R': { u[ n ]=value; break;}
  }
}

void Set_NeumannBC(double *u, const int n, const char letter){
  switch (letter) {
  case 'L': { u[ 1 ]=u[ 2 ]; break;}
  case 'R': { u[ n ]=u[n-1]; break;}
  }
}

void Manage_Comms(int phase, int domain, double **u) {
  MPI_Status status;
  // Communicate halo regions and impose BCs!
  //if (r== 0 ) Set_DirichletBC(*u,n,'L',0.0); // impose Dirichlet BC u[ 0 ] = 1.0
  //if (r== 0 ) Set_NeumannBC(*u,n,'L'); // impose Neumann BC : adiabatic condition 
  //if (r > 0 ) MPI_Send(*u + 1,1,MPI_DOUBLE,r-1,1,MPI_COMM_WORLD);         // send u[ 1 ] to   rank-1
  //if (r <p-1) MPI_Recv(*u+n+1,1,MPI_DOUBLE,r+1,1,MPI_COMM_WORLD,&status); // recv u[n+1] from rank+1
  //if (r <p-1) MPI_Send(*u+n  ,1,MPI_DOUBLE,r+1,2,MPI_COMM_WORLD);         // send u[ n ] to   rank+1
  //if (r > 0 ) MPI_Recv(*u    ,1,MPI_DOUBLE,r-1,2,MPI_COMM_WORLD,&status); // recv u[ 0 ] from rank-1
  //if (r==p-1) Set_NeumannBC(*u,n,'R'); // impose Neumann BC : adiabatic condition
  //if (r==p-1) Set_DirichletBC(*u,n,'R',1.0); // impose Dirichlet BC u[n+1] = 0.0
}

void Laplace2d(int nx, int ny, float *u,float *un){
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

void Call_Laplace(dmn domain, float **u, float **un){
  // Produce one iteration of the laplace operator
  Laplace2d(domain.nx,domain.ny,*u,*un);
}
