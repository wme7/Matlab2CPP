
#include "heat3d.h"

dmn Manage_Domain(int rank, int npcs){
  // allocate domain and its data
  dmn domain;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.nx = NX/SX;
  domain.ny = NY/SY;
  domain.nz = NZ/SZ;
  domain.size = domain.nx*domain.ny*domain.nz;
  
  // All process have by definition the same domain dimensions
  if ((NX*NY*NZ)%npcs != 0) {
    printf("Sorry, the domain size should be (%d*np) x (%d*1) x (%d*1) = NX*NY.\n",
	   domain.nx,domain.ny,domain.nz);
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  // Have process 0 print out some information.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n\n" );
    printf ("  C++/MPI version\n" );
    printf ("  Solve the 3D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf ("  Commence Simulation: procs rank %d out of %d cores"
	  " working with (%d +2) x (%d +2) x (%d +2) cells\n",
	  rank,npcs,domain.nx,domain.ny,domain.nz);

  return domain;
}

void Manage_Memory(int phase, dmn domain, double **g_u, double **h_u, double **h_un){
  if (phase==0) {
    // Allocate global domain on ROOT
    if (domain.rank==ROOT) *g_u=(double*)malloc(NX*NY*NZ*sizeof(double)); // only exist in ROOT!
    // Allocate local domains on MPI threats with 2 extra slots for halo regions
    *h_u =(double*)malloc((domain.nx+2)*(domain.ny+2)*(domain.nz+2)*sizeof(double));
    *h_un=(double*)malloc((domain.nx+2)*(domain.ny+2)*(domain.nz+2)*sizeof(double));
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
  int i, j, k, o; const int XY=NX*NY;
  switch (IC) {
  case 1: {
    for (k = 0; k < NZ; k++) {
      for (j = 0; j < NY; j++) {
	for (i = 0; i < NX; i++) {
	  // set all domain's cells equal to zero
	  o = i+NX*j+XY*k;  u0[o] = 0.0;
	  // set BCs in the domain 
	  if (k==0)    u0[o] = 1.0; // bottom
	  if (k==NZ-1) u0[o] = 1.0; // top
	}
      }
    }
    break;
  }
  case 2: {
    for (k = 0; k < NZ; k++) {
      for (j = 0; j < NY; j++) {
	for (i = 0; i < NX; i++) {
	  // set all domain's cells equal to zero
	  o = i+NX*j+XY*k;  
	  u0[o] = 1.0*exp(
			  -(DX*(i-NX/2))*(DX*(i-NX/2))/1.5
			  -(DY*(j-NY/2))*(DY*(j-NY/2))/1.5
			  -(DZ*(k-NZ/2))*(DZ*(k-NZ/2))/12);
	}
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
  const int XY=NX*NY;
  if (pFile != NULL) {
    for (int k = 0;k < NZ; k++) {
      for (int j = 0; j < NY; j++) {
	for (int i = 0; i < NX; i++) {      
	  fprintf(pFile, "%d\t %d\t %d\t %g\n",k,j,i,u[i+NX*j+XY*k]);
	}
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

void Laplace2d(const int nx, const int ny, const int nz, 
	       const double * __restrict__ u, double * __restrict__ un){
  // Using (i,j,k) = [i+N*j+M*N*k] indexes
  int i, j, k, o, n, s, e, w, t, b; 
  const int XY=nx*ny;
  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
	
	o = i+ (NX*j) + (XY*k); // node( j,i,k )      n  b
	n = (i==NX-1) ? o:o+NX; // node(j+1,i,k)      | /
	s = (i==0)    ? o:o-NX; // node(j-1,i,k)      |/
	e = (j==NY-1) ? o:o+1;  // node(j,i+1,k)  w---o---e
	w = (j==0)    ? o:o-1;  // node(j,i-1,k)     /|
	t = (k==NZ-1) ? o:o+XY; // node(j,i,k+1)    / |
	b = (k==0)    ? o:o-XY; // node(j,i,k-1)   t  s

	un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]) + KZ*(u[t]-2*u[o]+u[b]);
      }
    } 
  }
}

void Call_Laplace(dmn domain, double **u, double **un){
  // Produce one iteration of the laplace operator
  Laplace2d(domain.nx,domain.ny,domain.nz,*u,*un);
}
