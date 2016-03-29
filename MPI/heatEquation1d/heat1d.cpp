
#include "heat1d.h"

int Manage_Domain(int phase, int rank, int size){
  int nx;
  
  // Define the size for each slave
  if      (rank==    0  ) nx = NX;
  else if (rank < size-1) nx = ceil((float)NX/(size-1));
  else if (rank== size-1) nx = ceil((float)NX/(size-1)) - (NX%(size-1)); 

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
  int LX; if (size>1) LX = NX/(size-1); else LX = NX;
  if (phase==0) {
    // Allocate domain on host
    if (rank==0) {
      *h_u = (double*)malloc(NX*sizeof(double));
      *h_un= (double*)malloc(NX*sizeof(double));
    } else {
      *h_u = (double*)malloc((nx+2)*sizeof(double));
      *h_un= (double*)malloc((nx+2)*sizeof(double));
    }
  }
  if (phase==1) {
     // Distribute information from the main thread to the slaves
    if (rank==0) {
      MPI_Scatter(*h_u,)
      //MPI_Send(*h_u+0*LX,LX,MPI_DOUBLE,1,0,MPI_COMM_WORLD);  
      //MPI_Send(*h_u+1*LX,LX,MPI_DOUBLE,2,0,MPI_COMM_WORLD); 
      //MPI_Send(*h_u+2*LX,LX,MPI_DOUBLE,3,0,MPI_COMM_WORLD);
      //MPI_Send(*h_u+3*LX,LX,MPI_DOUBLE,4,0,MPI_COMM_WORLD);
      //MPI_Send(*h_u+4*LX,LX,MPI_DOUBLE,5,0,MPI_COMM_WORLD);
    } else {
      // This is a slave thread - prepare to recieve data.
      MPI_Recv(*h_u + 1 ,nx,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    } 
  }
  if (phase==2) {
    // Collect data from slaves to rank=0. 
    // 1 -> 0 
    if (rank == 1) MPI_Send(*h_u + 1 ,nx,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    if (rank == 0) MPI_Recv(*h_u+0*LX,LX,MPI_DOUBLE,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
    // 2 -> 0
    if (rank == 2) MPI_Send(*h_u + 1 ,nx,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    if (rank == 0) MPI_Recv(*h_u+1*LX,LX,MPI_DOUBLE,2,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (phase==3) {
    // Free the domain on host
    free(*h_u);
    free(*h_un);
  }
}

void Set_IC(double *u0){
  // Set initial condition in global domain

  const int IC=2;

  switch (IC) {
    case 1: {
      // Temperature at boundary 
      for (int i = 0; i < NX; i++) u0[i]=0.0;
      // Set Dirichlet boundary conditions in global domain
      u0[0]=0.0;  u0[NX-1]=1.0;
      break;
    }
    case 2: {
      // A square jump problem
      for (int i = 0; i < NX; i++) {if (i>0.3*NX && i<0.7*NX) u0[i]=1.0; else u0[i]=0.0;}
      // Set Dirichlet boundary conditions in global domain
      u0[0]=0.0;  u0[NX-1]=0.0;
      break;
    }
      // add another IC
    }
}

void Save_Results(double *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int i = 0; i < NX; i++) {
      fprintf(pFile, "%d\t %g\n",i,u[i]);
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

void Manage_Comms(int phase, int rank, int p, int n, double **u) {
  // communicate boundaries except with rank 0
  if (phase==1 && rank!=0) {    
    if (rank> 1 ) MPI_Send(*u + 1,1,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD);                  // send u[ 1 ] to   rank-1
    if (rank<p-1) MPI_Recv(*u+n+1,1,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE); // recv u[n+1] from rank+1
    if (rank<p-1) MPI_Send(*u+n  ,1,MPI_DOUBLE,rank+1,2,MPI_COMM_WORLD);                  // send u[ n ] to   rank+1
    if (rank> 1 ) MPI_Recv(*u    ,1,MPI_DOUBLE,rank-1,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE); // recv u[ 0 ] from rank-1
  }
  if (phase==2 && rank!=0) {
    if (rank> 1 ) MPI_Send(*u + 1,1,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD);                  // send u[ 1 ] to   rank-1
    if (rank<p-1) MPI_Recv(*u+n+1,1,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE); // recv u[n+1] from rank+1
    if (rank<p-1) MPI_Send(*u+n  ,1,MPI_DOUBLE,rank+1,2,MPI_COMM_WORLD);                  // send u[ n ] to   rank+1
    if (rank> 1 ) MPI_Recv(*u    ,1,MPI_DOUBLE,rank-1,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE); // recv u[ 0 ] from rank-1
  }
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


