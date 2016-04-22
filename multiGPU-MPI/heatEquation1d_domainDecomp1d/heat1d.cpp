
#include "heat1d.h"

void Call_IC(const int IC, real *u0){
  // Set initial condition in global domain
  switch (IC) {
    case 1: {
      // Uniform Temperature in the domain, temperature will be imposed at boundaries
      for (int i = 0; i < NX; i++) u0[i]=0.0;
      // Set Dirichlet boundary conditions in global domain as u0[0]=0.0;  u0[NX]=1.0; namely
      u0[0]=0.0; u0[NX]=1.0;
      break;
    }
    case 2: {
      // A square jump problem
      for (int i= 0; i < NX; i++) {if (i>0.3*NX && i<0.7*NX) u0[i]=1.0; else u0[i]=0.0;}
      // Set Neumann boundary conditions in global domain u0'[0]=0.0;  u0'[NX]=0.0;
      break;
    }
      // add another IC
  }
}

void Save_Results(real *u){
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

void Set_DirichletBC(real *u, const int n, const char letter, const real value){
  switch (letter) {
  case 'L': { u[ 1 ]=value; break;}
  case 'R': { u[ n ]=value; break;}
  }
}

void Set_NeumannBC(real *u, const int n, const char letter){
  switch (letter) {
  case 'L': { u[ 1 ]=u[ 2 ]; break;}
  case 'R': { u[ n ]=u[n-1]; break;}
  }
}

void Manage_Comms(dmn domain, real **u) {
  int n = domain.nx;
  MPI_Status status;
  // Communicate halo regions and impose BCs!
  //if (domain.rx==  0 ) Set_DirichletBC(*u,n,'L',0.0); // impose Dirichlet BC u[ 0 ] = 1.0
  if (domain.rx==  0 ) Set_NeumannBC(*u,n,'L'); // impose Neumann BC : adiabatic condition 
  if (domain.rx >  0 ) MPI_Send(*u + 1,1,MPI_CUSTOM_REAL,domain.rx-1,1,MPI_COMM_WORLD);         // send u[ 1 ] to   rank-1
  if (domain.rx <SX-1) MPI_Recv(*u+n+1,1,MPI_CUSTOM_REAL,domain.rx+1,1,MPI_COMM_WORLD,&status); // recv u[n+1] from rank+1
  if (domain.rx <SX-1) MPI_Send(*u+n  ,1,MPI_CUSTOM_REAL,domain.rx+1,2,MPI_COMM_WORLD);         // send u[ n ] to   rank+1
  if (domain.rx >  0 ) MPI_Recv(*u    ,1,MPI_CUSTOM_REAL,domain.rx-1,2,MPI_COMM_WORLD,&status); // recv u[ 0 ] from rank-1
  if (domain.rx==SX-1) Set_NeumannBC(*u,n,'R'); // impose Neumann BC : adiabatic condition
  //if (domain.rx==SX-1) Set_DirichletBC(*u,n,'R',1.0); // impose Dirichlet BC u[n+1] = 0.0
}
