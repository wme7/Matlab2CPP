
#include "heat2d.h"

void Manage_Memory(int phase, int tid, float **h_u, float **h_un){
  size_t domainSize;
  if (phase==0) {
    // Allocate whole domain variable h_u on host computer (master thread)
    domainSize= NX*NY*sizeof(float);
    *h_u = (float*)malloc(domainSize);
    *h_un= (float*)malloc(domainSize);
  }
  if (phase==1) {
    // Free the whole domain variables (master thread)
    free(*h_u);
    free(*h_un);
  }
}

void Set_IC(float *u0){
  // set all domain equal to zero
  for (int j = 0; j < NY; j++) {
    for (int i = 0; i < NX; i++) {
      int o = i+NX*j;

      u0[o] = 0.0;
      // but ...
      if (i==0)    u0[o] = 0.0;
      if (j==0)    u0[o] = 0.0; 
      if (i==NX-1) u0[o] = 1.0;
      if (j==NY-1) u0[o] = 1.0;
    }
  }
}

void Call_Init(float **u0){
  // Load the initial condition
  Set_IC(*u0);
}

void Laplace2d(float *u,float *un){
  // Using (i,j) = [i+N*j] indexes
  for (int j = 0; j < NY; j++) {
    for (int i = 0; i < NX; i++) {

        int o =  i + NX*j ; // node( j,i )     n
	int n = i+NX*(j+1); // node(j+1,i)     |
	int s = i+NX*(j-1); // node(j-1,i)  w--o--e
	int e = (i+1)+NX*j; // node(j,i+1)     |
	int w = (i-1)+NX*j; // node(j,i-1)     s

	// only update "interior" nodes
	if(i>0 && i<NX-1 && j>0 && j<NY-1) {
	  un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]);
	} else {
	  un[o] = u[o];
	}
    }
  } 
}

void Call_Laplace(float **u, float **un){
  // Produce one iteration of the laplace operator
  Laplace2d(*u,*un);
}

void Update_Domain(float *u, float *un){
  // Explicitly copy data arrays
  for (int j = 0; j < NY; j++) {
    for (int i = 0; i < NX; i++) {
      u[i+NX*j] = un[i+NX*j];
    }
  }
}

void Call_Update(float **u, float **un){
  // produce explicitly: u=un;
  Update_Domain(*u,*un);
}

void Save_Results(float *u){
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
