
#include "heat1d.h"

int Manage_Domain(int phase, int rank, int size){
  
  // All process have by definition the same domain size
  int nx = NX/size;
  if (NX%size != 0) {
    printf("Sorry, the domain size is should be %d*np = NX.\n",nx);
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  // Have process 0 print out some information.
  if (rank==0) {
    printf ("HEAT_MPI:\n\n" );
    printf ("  C++/MPI version\n" );
    printf ("  Solve the 1D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf ("  Commence Simulation: processor rank %d out of %d processors"
	  " working with %d cells\n",rank,size,nx);

  // return the nx value
  return nx;
}

void Manage_Memory(int phase, int rank, int size, int nx, double **h_u, double **h_un){
  if (phase==0) {
    // Allocate domain on host with 2 extra slots for halo region
    *h_u = (double*)malloc((nx+2)*sizeof(double));
    *h_un= (double*)malloc((nx+2)*sizeof(double));
  }
  if (phase==1) {
    // Free the domain on host
    free(*h_u);
    free(*h_un);
  }
}

void Call_IC(const int IC, int rank, int size, int nx, double *u0){
  // Set initial condition in global domain
  switch (IC) {
    case 0: {
      // Testing
      if (rank==0) {for (int i = 0; i < nx; i++) {u0[i+1]=0.2;}}
      if (rank==1) {for (int i = 0; i < nx; i++) {u0[i+1]=0.6;}}
      if (rank==2) {for (int i = 0; i < nx; i++) {u0[i+1]=1.0;}}
      break;
    }
    case 1: {
      // Uniform Temperature in the domain, temperature will be imposed at boundaries
      for (int i=rank; i<(rank+1)*nx; i++) {u0[i+1]=0.0;}
      // Set Dirichlet boundary conditions in global domain
      //u0[0]=0.0;  u0[NX-1]=1.0;
      break;
    }
    case 2: {
      // A square jump problem
      for (int i=rank; i<(rank+1)*nx; i++) {if (i>0.3*NX && i<0.7*NX) u0[i+1]=1.0; else u0[i+1]=0.0;}
      // Set Dirichlet boundary conditions in global domain
      //u0[0]=0.0;  u0[NX-1]=0.0;
      break;
    }
      // add another IC
  }
}

void Save_Results(int nx,double *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int i = 0; i < nx+2; i++) {
      fprintf(pFile, "%d\t %g\n",i,u[i]);
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

void Manage_Comms(int r, int p, int n, double **u) {
  MPI_Status status;
  // communicate boundaries 
  //if (r== 0 ) // impose dirichlet BC
  if (r > 0 ) MPI_Send(*u + 1,1,MPI_DOUBLE,r-1,1,MPI_COMM_WORLD);         // send u[ 1 ] to   rank-1
  if (r <p-1) MPI_Recv(*u+n+1,1,MPI_DOUBLE,r+1,1,MPI_COMM_WORLD,&status); // recv u[n+1] from rank+1
  if (r <p-1) MPI_Send(*u+n  ,1,MPI_DOUBLE,r+1,2,MPI_COMM_WORLD);         // send u[ n ] to   rank+1
  if (r > 0 ) MPI_Recv(*u    ,1,MPI_DOUBLE,r-1,2,MPI_COMM_WORLD,&status); // recv u[ 0 ] from rank-1
  //if (r==p-1) // impose dirichlet BC
}

/*
void Laplace1d(const double * __restrict__ u, double * __restrict__ un){
  int i, o, r, l;
  // perform laplace operator
  for (i = 0; i < n; i++) {

     o =   i  ; // node( j,i ) 
     r = (i+1); // node(j-1,i)  l--o--r
     l = (i-1); // node(j,i-1)  
     
     // only update "interior" nodes
     if(i>0 && i<n-1) {
       un[o] = u[o] + KX*(u[r]-2*u[o]+u[l]);
     } else {
       un[o] = u[o];
     }
  } 
}

void Call_Laplace(int rank, double **u, double **un){
  // Produce one iteration of the laplace operator
  switch (rank) {
  case 0: {do nothing break;}
  default: {Laplace1d(*u,*un); break;}
  }
}
*/


