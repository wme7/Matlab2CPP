
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEBUG 1 // Display all error messages
#define NX 1024 // number of cells in the x-direction
#define L 1.0 // domain length
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 0.02 // tEnd, output time
#define DX (L/NX) // dx, cell size
#define DT (1/(2*C*(1/DX/DX))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define NO_STEPS (TEND/DT) // No. of time steps
#define XGRID 8 // No. of subdomains in the x-direction
#define NO_GPU XGRID // No. of GPUs and OMP threads
#define SNX (NX/XGRID) // subregion size
#define PI 3.1415926535897932f

/* Declare functions */
void Manage_Memory(int phase, int tid, float **h_u, float **t_u, float **t_un);
void Manage_Comms(int phase, int tid, float **t_u, float **t_un);

void Call_Init(float **h_u);
void Call_GPU_Init(int tid,float **t_u);

void Call_Laplace(int tid, float **t_u,float **t_un);
void Call_Update(int tid, float **h_u,float **t_un);
void Save_Results(float *h_u);
