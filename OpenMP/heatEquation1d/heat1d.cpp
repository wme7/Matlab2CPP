
#include "heat1d.h"

void Manage_Memory(int phase, int tid, float **h_u, float **t_u, float **t_un){
  if (phase==0) {
    // Allocate domain variable on host (master thread)
    *h_u = (float*)malloc(NX*sizeof(float));
   }
  if (phase==1) {
    // Allocate subdomain variables on host (All Threads)
    *t_u = (float*)malloc((SNX+2)*sizeof(float));
    *t_un= (float*)malloc((SNX+2)*sizeof(float));
   }
  if (phase==2) {
    // Free the whole domain variables (master thread)
    free(*t_u);
    free(*t_un);
  }
  if (phase==3) {
    // Free the whole domain variables (master thread)
    free(*h_u);
  }
}

void Manage_Comms(int tid, float *h_u, float *t_u){
  // Communicate boundaries
  if (tid==0) h_u[tid*SNX] = t_u[1]; 
  if (tid==1) h_u[tid*SNX] = t_u[1];
  if (tid==2) h_u[tid*SNX] = t_u[1];
  if (tid==3) h_u[tid*SNX] = t_u[1];
}

void Set_IC(int tid, float *u0){
  // Set domain initial condition 
  if (tid==0) {
    for (int i = 0; i < SNX; i++) {
      u0[i+1] = 0.25;
      // but ...
      if (i==0)     u0[i+1] = 0.0;
    }
  }
  if (tid>0 && tid<OMP_THREADS-1) {
    for (int i = 0; i < SNX; i++) {
      u0[i+1] = 0.50;
    } 
  }
  if (tid==OMP_THREADS-1) {
    for (int i = 0; i < SNX; i++) {
      u0[i+1] = 0.75;
      // but ...
      if (i==SNX-1) u0[i+1] = 1.0;
    }
  }
}

void Call_Init(int tid, float **u0){
  // Load the initial condition
  Set_IC(tid,*u0);
}

void Laplace1d(float *u,float *un){
  // Using (i,j) = [i+N*j] indexes
  int o, l, r;
  for (int i = 0; i < SNX+2; i++) {

    o =   i  ; // node( j,i ) 
    r = (i+1); // node(j-1,i)  l--o--r
    l = (i-1); // node(j,i-1) 

    // only update "interior" nodes
    if(i>0 && i<SNX+1) {
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

void Update_Domain(int tid, float *h_u, float *t_u){
  // Explicitly copy data arrays
  if (DEBUG) printf("Copying thread data into the whole domain (thread %d)\n",tid); 
  for (int i = 0; i < SNX; i++) {
    h_u[i+tid*SNX] = t_u[i+1];
  }
}

void Call_Update(int tid, float **h_u, float **t_u){
  // produce explicitly: h_u = t_u
  Update_Domain(tid,*h_u,*t_u);
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
