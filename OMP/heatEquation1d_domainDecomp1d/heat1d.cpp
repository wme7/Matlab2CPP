
#include "heat1d.h"

void Manage_Memory(int phase, int tid, float **h_u, float **t_u, float **t_un){
  if (phase==0) {
    // Allocate domain variable on host (master thread)
    *h_u = (float*)malloc((NX+2)*sizeof(float));
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

void Boundaries(int tid, float *h_u, float *t_u){
  // Communicate BCs from local thread to global domain
  h_u[ 1 +tid*SNX] = t_u[ 1 ];
  h_u[SNX+tid*SNX] = t_u[SNX];
  // Communicate BCs from global domain to local thread
  t_u[  0  ] = h_u[  0  +tid*SNX];
  t_u[SNX+1] = h_u[SNX+1+tid*SNX];
}

void Manage_Comms(int tid, float **h_u, float **t_u){
  // Manage boundary comunications
  Boundaries(tid,*h_u,*t_u);
}


void Set_IC(int phase, int tid, float *u0, float *ut0){
  if (phase==0) {
    // Set initial condition in global domain
    for (int i = 1; i < NX+1; i++) {u0[i] = 0.0;}  u0[0]=0.0;  u0[NX+1]=1.0;
  }
  if (phase==1) {
    // Set domain initial condition in local threads
    for (int i = 0; i < SNX+2; i++) {
      ut0[i] = 0.0;
    }
  }
}

void Call_Init(int phase, int tid, float **u0, float **ut0){
  // Load the initial condition
  Set_IC(phase,tid,*u0,*ut0);
}

void Laplace1d(int tid, float *u,float *un){
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

void Call_Laplace(int tid, float **u, float **un){
  // Produce one iteration of the laplace operator
  Laplace1d(tid,*u,*un);
}

void Update_Domain(int tid, float *h_u, float *t_u){
  // Explicitly copy data arrays
  if (DEBUG) printf("Copying thread data into the whole domain (thread %d)\n",tid); 
  for (int i = 0; i < SNX; i++) {
    h_u[i+1+tid*SNX] = t_u[i+1];
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
    for (int i = 0; i < NX+2; i++) {
      fprintf(pFile, "%d\t %g\n",i,u[i]);
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}
