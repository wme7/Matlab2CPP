
#include "heat2d.h"

void Manage_Memory(int phase, int tid, float **h_u, float **t_u, float **t_un){
  if (phase==0) {
    // Allocate domain variable on host (master thread)
    *h_u = (float*)malloc(NY*NX*sizeof(float));
   }
  if (phase==1) {
    // Allocate subdomain variables on host (All Threads)
    *t_u = (float*)malloc(SNY*SNX*sizeof(float));
    *t_un= (float*)malloc(SNY*SNX*sizeof(float));
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

void Manage_Comms(int phase, int tid, float *h_u, float *t_u){
}

void Set_IC(int tid, float *u0){
  // Set domain initial condition 
  if (tid==0) {
      for (int j = 1; j < SNY-1; j++) {
	for (int i = 1; i < SNX-1; i++) {
	  u0[i+SNX*j] = 0.0;
	  // but ...
	  if (i==1)     u0[i+SNX*j] = 0.0;
	  if (j==1)     u0[i+SNX*j] = 0.0; 
	  if (j==SNY-2) u0[i+SNX*j] = 1.0;
	}
      }
  }
  if (tid>0 && tid<OMP_THREADS-1) {
    for (int j = 1; j < SNY-1; j++) {
	for (int i = 1; i < SNX-1; i++) {
	  u0[i+SNX*j] = 0.0;
	  // but ...
	  if (j==1)     u0[i+SNX*j] = 0.0; 
	  if (j==SNY-2) u0[i+SNX*j] = 1.0;
	}
    } 
  }
  if (tid==OMP_THREADS-1) {
    for (int j = 0; j < SNY; j++) {
	for (int i = 0; i < SNX; i++) {
	  u0[i+SNX*j] = 0.0;
	  // but ...
	  if (j==1)     u0[i+SNX*j] = 0.0; 
	  if (i==SNX-2) u0[i+SNX*j] = 1.0;
	  if (j==SNY-2) u0[i+SNX*j] = 1.0;
	}
    }
  }
}

void Call_Init(int tid, float **u0){
  // Load the initial condition
  Set_IC(tid,*u0);
}

void Laplace2d(float *u,float *un){
  // Using (i,j) = [i+N*j] indexes
  int o, n, s, e, w;
  for (int j = 0; j < SNY; j++) {
    for (int i = 0; i < SNX; i++) {

        o =  i + SNX*j ; // node( j,i )     n
	n = i+SNX*(j+1); // node(j+1,i)     |
	s = i+SNX*(j-1); // node(j-1,i)  w--o--e
	e = (i+1)+SNX*j; // node(j,i+1)     |
	w = (i-1)+SNX*j; // node(j,i-1)     s

	// only update "interior" nodes
	if(i>0 && i<SNX-1 && j>0 && j<SNY-1) {
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

void Update_Domain(int tid, float *h_u, float *t_u){
  // Explicitly copy data arrays
  if (DEBUG) printf("Copying thread data into the whole domain (thread %d)\n",tid);
  for (int j = 1; j < SNY-1; j++) {
    for (int i = 1; i < SNX-1; i++) {
      h_u[((i-1)+tid*(SNX-2))+NX*(j-1)] = t_u[i+SNX*j];
      
    }
  }
  //  For a grid of subdomains: 
  //  if (tid==0) {
  //  for (int j = 0; j < SNY; j++) {
  //      for (int i = 0; i < SNX; i++) {
  //	h_u[i+NX*j] = t_u[i+SNX*j];
  //      }
  //    }
  //  }
  //  if (tid>0 && tid<TOT) {
  //    for (int j = 0; j < SNY; j++) {
  //      for (int i = 0; i < SNX; i++) {
  //	h_u[(i+tid*SNX)+NX*j] = t_u[i+SNX*j];
  //      }
  //    }
  //  }
  //  if (tid===OMP_THREADS-1) {
  //    for (int j = 0; j < SNY; j++) {
  //      for (int i = 0; i < SNX; i++) {
  //	h_u[(i+TOT*SNX)+NX*j] = t_u[i+SNX*j];
  //      }
  //    }
  //  }
}

void Call_Update(int tid, float **h_u, float **t_u){
  // produce explicitly: h_u = t_u
  Update_Domain(tid,*h_u,*t_u);
}

void Save_Results(float *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int j = 0; j < NY; j++) {
      for (int i = 0; i < NX; i++) {      
	fprintf(pFile, "%d\t %d\t %g\n",i,j,u[i+NX*j]);
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}
