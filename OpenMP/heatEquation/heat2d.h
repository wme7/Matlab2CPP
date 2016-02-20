
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEBUG 0 // Display all error messages
#define NX 150 // number of cells in the x-direction
#define NY 100 // number of cells in the y-direction
#define L 1.5 // domain length
#define H 1.0 // domain height
#define DX (L/N) // dx, cell size
#define DY (H/M) // dy, cell size
#define DT (0.1*DX*DX*DY*DY) // dt, time step size
#define NO_STEPS 1000 // No. of time steps
#define OMP_THREADS 3 // No. of OMP threads
#define PI 3.1415926535897932f

/* Declare functions */
void Manage_Memory(int phase, int tid, float **h_u, float **t_u);
void Call_Init(int tid, float **t_u);
//void Call_Laplace(int tid, float **h_u,float **t_u);
void Call_Update(int tid, float **h_u,float **t_u);
void Save_Results(float *h_u);
