
#include "heat2d.h"

void Manage_Memory(int phase, int tid, float **h_u, float **t_u, float **t_un){
  if (phase==0) {
    // Allocate domain variable on host (master thread)
    *h_u = (float*)malloc(( NY+2)*( NX+2)*sizeof(float));
   }
  if (phase==1) {
    // Allocate subdomain variables on host (All Threads)
    *t_u = (float*)malloc((SNY+2)*(SNX+2)*sizeof(float));
    *t_un= (float*)malloc((SNY+2)*(SNX+2)*sizeof(float));
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

void Boundaries(int phase, int tid, float *h_u, float *t_u){
  if (phase==1) {
    // Communicate BCs from local thread to global domain
    for (int j = 0; j < SNY; j++) {
      h_u[ 1 +tid*SNX+(NX+2)*(j+1)] = t_u[ 1 +(SNX+2)*(j+1)]; // left
      h_u[SNX+tid*SNX+(NX+2)*(j+1)] = t_u[SNX+(SNX+2)*(j+1)]; // right
    }
    for (int i = 0; i < SNX; i++) {
      h_u[i+1+tid*SNX+(NX+2)* 1 ] = t_u[i+1+(SNX+2)* 1 ]; // up
      h_u[i+1+tid*SNX+(NX+2)*SNY] = t_u[i+1+(SNX+2)*SNY]; // down
    }
  }
  if (phase==2) {
    // Communicate BCs from global domain to local thread
    for (int j = 0; j < SNY; j++) {
      t_u[  0  +(SNX+2)*(j+1)] = h_u[  0  +tid*SNX+(NX+2)*(j+1)]; // left
      t_u[SNX+1+(SNX+2)*(j+1)] = h_u[SNX+1+tid*SNX+(NX+2)*(j+1)]; // right
    }
    for (int i = 0; i < SNX; i++) {
      t_u[i+1+(SNX+2)*   0   ] = h_u[i+1+tid*SNX+(NX+2)*   0   ]; // up 
      t_u[i+1+(SNX+2)*(SNY+1)] = h_u[i+1+tid*SNX+(NX+2)*(SNY+1)]; // down
    }
  }
  if (phase==3) {
    // Explicitly copy data arrays
    if (DEBUG) printf("Copying thread data into the whole domain (thread %d)\n",tid);
    for (int j = 0; j < SNY; j++) {
      for (int i = 0; i < SNX; i++) {
	h_u[i+1+tid*SNX+(NX+2)*(j+1)] = t_u[i+1+(SNX+2)*(j+1)];
      }
    }
  }
}

void Manage_Comms(int phase, int tid, float **h_u, float **t_u){
  // Manage boundary comunications
  Boundaries(phase,tid,*h_u,*t_u);
}

void Set_IC_globalDomain(float *u0){
  // Set Dirichlet boundary conditions in global domain
  for (int i = 0; i < NX+2; i++) u0[   i  +(NX+2)*   0  ]=0.0; // down  
  for (int j = 0; j < NY+2; j++) u0[   0  +(NX+2)*   j  ]=0.0; // left
  for (int i = 0; i < NX+2; i++) u0[   i  +(NX+2)*(NY+1)]=1.0; // up
  for (int j = 0; j < NY+2; j++) u0[(NX+1)+(NX+2)*   j  ]=1.0; // right
}

void Call_Init_globalDomain(float **u0){
  // Set Dirichlet boundary conditions in global domain
  Set_IC_globalDomain(*u0);
}

void Set_IC_localDomain(int tid, float *u){
  // Set domain initial condition 
  for (int j = 0; j < SNY+2; j++) {
    for (int i = 0; i < SNX+2; i++) {
      int o=i+(SNX+2)*j; u[o] = 0.0;

      if (i>0 && i<SNX+1 && j>0 && j<SNY+1) {
	switch (tid) {
	  case 0: u[o] = 0.10; break;
	  case 1: u[o] = 0.25; break;
	  case 2: u[o] = 0.40; break;
	  case 3: u[o] = 0.50; break;
	  case 4: u[o] = 0.75; break;
	  case 5: u[o] = 0.90; break;
	  }
      }
    }
  }
}

void Call_Init_localDomain(int tid, float **u0){
  // Load the initial condition
  Set_IC_localDomain(tid,*u0);
}

void Laplace2d(float *u,float *un){
  // Using (i,j) = [i+N*j] indexes
  int o, n, s, e, w;
  for (int j = 0; j < SNY+2; j++) {
    for (int i = 0; i < SNX+2; i++) {

      o =  i + (SNX+2)*j ; // node( j,i )     n
      n = i+(SNX+2)*(j+1); // node(j+1,i)     |
      s = i+(SNX+2)*(j-1); // node(j-1,i)  w--o--e
      e = (i+1)+(SNX+2)*j; // node(j,i+1)     |
      w = (i-1)+(SNX+2)*j; // node(j,i-1)     s

      // only update "interior" nodes
      if(i>0 && i<SNX+1 && j>0 && j<SNY+1) {
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

void Save_Results(float *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int j = 0; j < NY+2; j++) {
      for (int i = 0; i < NX+2; i++) {
	fprintf(pFile, "%d\t %d\t %g\n",i,j,u[i+(NX+2)*j]);
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

void Save_Results_Tid(int tid, float *u){
  // print result to txt file
  if (tid==0) {
    FILE *pFile = fopen("result0.txt", "w");
    if (pFile != NULL) {
      for (int j = 0; j < SNY+2; j++) {
	for (int i = 0; i < SNX+2; i++) {
	  fprintf(pFile, "%d\t %d\t %g\n",i,j,u[i+(SNX+2)*j]);
	}
      }
      fclose(pFile);
    } else {
      printf("Unable to save to file\n");
    }
  }
  if (tid==1) {
    FILE *pFile = fopen("result1.txt", "w");
    if (pFile != NULL) {
      for (int j = 0; j < SNY+2; j++) {
	for (int i = 0; i < SNX+2; i++) {
	  fprintf(pFile, "%d\t %d\t %g\n",i,j,u[i+(SNX+2)*j]);
	}
      }
      fclose(pFile);
    } else {
      printf("Unable to save to file\n");
    }
  }
}
