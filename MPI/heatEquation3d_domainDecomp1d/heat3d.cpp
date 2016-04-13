
#include "heat3d.h"

dmn Manage_Domain(int rank, int npcs){
  // allocate sub-domain for a one-dimensional domain decomposition in the Z-direction
  dmn domain;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.nx = NX/1;
  domain.ny = NY/1;
  domain.nz = NZ/npcs;
  domain.size = domain.nx*domain.ny*domain.nz;
  
  // All process have by definition the same domain dimensions
  if ( NZ%npcs != 0) {
    printf("Sorry, the domain size should be (%d*1) x (%d*1) x (%d*np) = NX*NY*NZ.\n",
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
	  " working with (%d +0) x (%d +0) x (%d +2) cells\n",
	  rank,npcs,domain.nx,domain.ny,domain.nz);

  return domain;
}

void Manage_Memory(int phase, dmn domain, double **g_u, double **h_u, double **h_un){
  if (phase==0) {
    // Allocate global domain on ROOT
    if (domain.rank==ROOT) *g_u=(double*)malloc(NX*NY*NZ*sizeof(double)); // only exist in ROOT!
    // Allocate local domains on MPI threats with 2 extra slots for halo regions
    *h_u =(double*)malloc((domain.nx+0)*(domain.ny+0)*(domain.nz+2*R)*sizeof(double));
    *h_un=(double*)malloc((domain.nx+0)*(domain.ny+0)*(domain.nz+2*R)*sizeof(double));
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

void Set_NeumannBC(double *u, const int l, const char letter){
  int XY=NX*NY, i, j;
  switch (letter) { 
  case 'B': { 
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
	u[i+NX*j+XY*l-1]=u[i+NX*j+XY*l];
      }
    }
    break;
  }
  case 'T': { 
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
	u[i+NX*j+XY*l+1]=u[i+NX*j+XY*l];
      }
    }
    break;
  }
  }
}

void Manage_Comms(dmn domain, double **u) {
  MPI_Status status; 
  MPI_Request rqSendUp, rqSendDown, rqRecvUp, rqRecvDown;
  const int r = domain.rank;
  const int p = domain.npcs;
  const int nm= domain.nx*domain.ny; // n*m layer
  const int l = domain.nz;
  
// Impose BCs!
  if (r== 0 ) Set_NeumannBC(*u,1,'B'); // impose Dirichlet BC u[row  1 ]
  if (r==p-1) Set_NeumannBC(*u,l,'T'); // impose Dirichlet BC u[row L-1]

  // Communicate halo regions
  if (r <p-1) {
    MPI_Isend(*u+nm*l    ,nm,MPI_DOUBLE,r+1,2,MPI_COMM_WORLD,&rqSendDown); // send u[layerL-1] to   rank+1
    MPI_Irecv(*u+nm*(l+R),nm,MPI_DOUBLE,r+1,1,MPI_COMM_WORLD,&rqRecvUp  ); // recv u[layer L ] from rank+1
  }
  if (r > 0 ) {
    MPI_Isend(*u+nm      ,nm,MPI_DOUBLE,r-1,1,MPI_COMM_WORLD,&rqSendUp  ); // send u[layer 1 ] to   rank-1
    MPI_Irecv(*u         ,nm,MPI_DOUBLE,r-1,2,MPI_COMM_WORLD,&rqRecvDown); // recv u[layer 0 ] from rank-1
  }

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

void Laplace2d(const int nz, const double * __restrict__ u, double * __restrict__ un){
  // Using (i,j,k) = [i+N*j+M*N*k] indexes
  int i, j, k, o, n, s, e, w, t, b; 
  const int XY=NX*NY;
  for (k = 0; k < nz; k++) {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
	
	o = i+ (NX*j) + (XY*k); // node( j,i,k )      n  b
	n = (i==NX-1) ? o:o+NX; // node(j+1,i,k)      | /
	s = (i==0)    ? o:o-NX; // node(j-1,i,k)      |/
	e = (j==NY-1) ? o:o+1;  // node(j,i+1,k)  w---o---e
	w = (j==0)    ? o:o-1;  // node(j,i-1,k)     /|
	t =               o+XY; // node(j,i,k+1)    / |
	b =               o-XY; // node(j,i,k-1)   t  s

	// only update "interior" nodes
	if (k>0 && k<nz-1) {
	  un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]) + KZ*(u[t]-2*u[o]+u[b]);
	} else {
	  un[o] = u[o];
	}
      }
    } 
  }
}

void Call_Laplace(dmn domain, double **u, double **un){
  // Produce one iteration of the laplace operator
  Laplace2d(domain.nz+2*R,*u,*un);
}
