
#include "heat2d.h"

void Manage_Memory(int phase, int tid, float **h_u, float **t_u){
  size_t domainSize;
  if (phase==0) {
    // Allocate global domain variable (h_u) on host (master thread)
    domainSize= NY*NX*sizeof(float);
    *h_u = (float*)malloc(domainSize);
   }
  if (phase==1) {
    // Allocate local domain (t_u) on host (All Threads)
    int SN = NX/OMP_THREADS;    
    domainSize= NY*SN*sizeof(float);
    *t_u = (float*)malloc(domainSize);
   }
  if (phase==2) {
    // Free the whole domain variables (master thread)
    free(*h_u);
  }
}

void Set_IC(int tid, float *t_u){
  // Set all domain initial condition and boundaries
  int SN = NX/OMP_THREADS;
  int OT = OMP_THREADS-1;
  if (tid==1) {
      for (int j = 0; j < NY; j++) {
	for (int i = 0; i < SN; i++) {
	  //if (i==0)         t_u[i+SN*j] = 1.0; 
	  //else if (j==0)    t_u[i+SN*j] = 1.0; 
	  //else if (j==NY-1) t_u[i+SN*j] = 0.0;
	  //else              t_u[i+SN*j] = 0.0;
	   t_u[i+SN*j] = 1.0;
	}
      }
  }
  if (tid!=1 && tid!=OT) {
    for (int j = 0; j < NY; j++) {
	for (int i = 0; i < SN; i++) {
	  //if (j==0)         t_u[i+SN*j] = 1.0; 
	  //else if (j==NY-1) t_u[i+SN*j] = 0.0;
	  //else              t_u[i+SN*j] = 0.0;
	  t_u[i+SN*j] = 2.0;
	}
    } 
  }
  if (tid==OT) {
    for (int j = 0; j < NY; j++) {
	for (int i = 0; i < SN; i++) {
	  //if (i==SN-1)      t_u[i+SN*j] = 0.0;
	  //else if (j==0)    t_u[i+SN*j] = 1.0; 
	  //else if (j==NY-1) t_u[i+SN*j] = 0.0;
	  //else              t_u[i+SN*j] = 0.0;
	  t_u[i+SN*j] = 3.0;
	}
    }
  }
}

void Call_Init(int tid, float **t_u){
  // Load the initial condition
  Set_IC(tid,*t_u);
}

//void Jacobi_Method(int tid, float *u, float *t){
  // Compute explicitly the jacobi stencil
  //
  // In this implementation all cells are assigned with a single index k.
  // Knowing that c/c++ use row-major order, we have:
  //
  //         1          2  ...    N
  //       N+1        N+2  ...  2*N
  //    2 *N+1     2 *N+3  ...  3*N
  //       ...        ...  ...  ...
  // (M-1)*N+1  (M-1)*N+2  ...  M*N
  //
  // Therefore, the neighbors of a selected k-cell are
  //
  //         k-N
  //          |
  //   k-1 -- k -- k+1
  //          |
  //         k+N
  //
  // cells on the lower boundary satisfy:
  //   1 <= k <= N
  // cells on the upper boundary satisfy:
  //   (M-1)*(N+1) <= k <= (M*N)
  // cells on the left boundary satisfy:
  //   mod( k, N ) = 1
  // cells on the right boundary satisfy:
  //   mod( k, N ) = 0
  //
  // If we number rows from bottom j = 0 to top j = M-1
  // and columns from left i = 0 to right i = N-1, we have
  //
  //   k = i + N*j
  //
  // Using single k-index
  // for (int k = 0; k < N*M; k++) {
  //  if ( k>N  &&  k<(M-1)*(N+1)  &&  k%N==1  &&  k%N==0 ) {
  //    un[] = u[];      
  //  }
  // }
  //
  // Using (i,j) = [i+N*j] index 
//  for (int j = 1; j < NY-1; j++) {
//    for (int i = 1; i < NX-1; i++) {
//      t[i+NX*j] = 0.25f*(u[i+NX*(j+1)] + u[i+NX*(j-1)] + u[(i+1)+NX*j] + u[(i-1)+NX*j]);
//    }
//  }
//}

//void Call_Laplace(float **h_u, float **t_u){
  // Produce one iteration of the laplace operator
//  Jacobi_Method(*h_u,*t_u);
//}

void Update_Domain(int tid, float *h_u, float *t_u){
  // Explicitly copy data arrays
  if (DEBUG) printf("Copying thread data into the whole domain (thread %d)\n",tid);
  int SN = NX/OMP_THREADS;
  int OT = OMP_THREADS-1;
  if (tid==0) {
    for (int j = 1; j < NY-1; j++) {
      for (int i = 1; i < SN-1; i++) {
	h_u[i+NX*j] = t_u[i+SN*j];
      }
    }
  }
  if (tid!=0 && tid!=OT) {
    for (int j = 1; j < NY-1; j++) {
      for (int i = 1; i < SN-1; i++) {
	h_u[(i+SN)+NX*j] = t_u[i+SN*j];
      }
    }
  }
  if (tid==OT) {
    for (int j = 1; j < NY-1; j++) {
      for (int i = 1; i < SN-1; i++) {
	h_u[(i+2*SN)+NX*j] = t_u[i+SN*j];
      }
    }
  }
}

void Call_Update(int tid, float **h_u, float **t_u){
  // produce explicitly: h_u = t_u
  Update_Domain(tid,*h_u,*t_u);
}

void Save_Results(float *u){
  // print result to txt file
  FILE *pFile = fopen("result", "w");
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
