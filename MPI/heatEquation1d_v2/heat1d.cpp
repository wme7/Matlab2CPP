
#include "heat1d.h"

int Manage_Domain(int phase, int rank, int size){
  int nx;
  
  // Define the size for each slave
  if      (rank==    0  ) nx = NX;
  else if (rank < size-1) nx = ceil((float)NX/(size-1)) +2;
  else if (rank== size-1) nx = ceil((float)NX/(size-1)) - (NX%(size-1)) +2; 

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
  size_t global= NX*sizeof(double);
  size_t  local= nx*sizeof(double);
  if (phase==0) {
    // Allocate domain on host
    if (rank==0) {
      *h_u = (double*)malloc(global);
      *h_un= (double*)malloc(global);
    } else {
      *h_u = (double*)malloc( local);
      *h_un= (double*)malloc( local);
    }
  }
  if (phase==1) {
     // Distribute information from the main thread to the slaves
    if (rank==0) {
      MPI_Send(*h_u+0*NX/2, NX/2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);  
      MPI_Send(*h_u+1*NX/2, NX/2, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);  
    } else {
      // This is a slave thread - prepare to recieve data.
      MPI_Recv(*h_u+1, nx, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } 
  }
  if (phase==2) {
    // Collect data from slaves if rank=0. Otherwise, send it.
    // 1 --> 0 First
    if (rank == 1) {
      MPI_Send(*h_u+1, nx, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else if (rank == 0) {
      MPI_Recv(*h_u+0*NX/2, NX/2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // Once rank 0 is ready, we will continue.
    MPI_Barrier(MPI_COMM_WORLD);
    // 2 --> 0 Now
    if (rank == 2) {
      MPI_Send(*h_u+1, nx, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else if (rank == 0) {
      MPI_Recv(*h_u+1*NX/2, NX/2, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // Add one final barrier for good measure.
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

void Manage_Comms(int phase, int rank, int size, int nx, double **h_u) {
  if (phase==1) {
    // Each rank (except rank 0) passes the right end of its data (element NP) to the 1st element
    // of its right hand side neighbour region
    // |   1 -----> 2   |
    // |	|	|
  if (rank == 1) {
    // This rank is sending one value
    MPI_Send(*h_a+NP, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
  } else  if (rank == 2) {
    // This rank is recieving one value
    MPI_Recv(*h_a, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  MPI_Barrier(MPI_COMM_WORLD); // Make sure we are all ready.
  
  }
  if (phase==2) {
    // Each rank (except rank 0) passes the left end of its data 
    // to the LAST element of its left hand side neighbour region
    // |   1 <----- 2   |
    // |	|	|
    if (rank == 2) {
      // These ranks are passing 1 value
      MPI_Send(*h_a+1, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
      // These ranks are recieving
      MPI_Recv(*h_a+NP+1, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
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
