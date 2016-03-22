
#include "heat1d.h"

void Manage_Memory(int phase, float **h_u, float **h_un){
  size_t global= NX*sizeof(float);
  if (phase==0) {
    // Allocate domain on host
    *h_u = (float*)malloc(global);
    *h_un= (float*)malloc(global);
   }
  if (phase==1) {
    // Free the domain on host
    free(*h_u);
    free(*h_un);
  }
}

void Set_IC(float *u0){
  // Set initial condition in global domain
  for (int i = 0; i < NX; i++) u0[i]=0.0;
  // Set Dirichlet boundary conditions in global domain
  u0[0]=0.0;  u0[NX-1]=1.0;
}

void Call_Init(float **u0){
  // Load the initial condition
  Set_IC(*u0);
}

void Laplace1d(const float * __restrict__ u, float * __restrict__ un){
  int i, o, r, l;
  // perform laplace operator
  for (i = 0; i < NX; i++) {

     o =   i  ; // node( j,i ) 
     r = (i+1); // node(j-1,i)  l--o--r
     l = (i-1); // node(j,i-1)  
     
     // only update "interior" nodes
     if(i>0 && i<NX-1) {
       un[o] = u[o] + KX*(u[r]-2*u[o]+u[l]);
     } else {
       un[o] = u[o];
     }
  } 
}

void Call_Laplace(float **u, float **un){
  // Produce one iteration of the laplace operator
  Laplace1d(*u,*un);
}

void Save_Results(float *u){
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
