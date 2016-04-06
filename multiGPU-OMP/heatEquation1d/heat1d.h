
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEBUG 0 // Display all error messages
#define NX 1024 // number of cells in the x-direction
#define L 1.0 // domain length
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 0.02 // tEnd, output time
#define DX (L/NX) // dx, cell size
#define DT (1/(2*C*(1/DX/DX))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define NO_STEPS (TEND/DT) // No. of time steps
#define NO_GPU 2 // No. of GPUs and OMP threads
#define XGRID NO_GPU // No. of subdomains in the x-direction
#define SNX (NX/XGRID) // subregaion size
#define PI 3.1415926535897932f

/* Declare functions */
void Manage_Memory(int phase, int tid, float **h_u, float **d_u, float **d_un);
void Manage_Comms(int phase, int tid, float **h_u, float **d_un);

void Call_Init(float **h_u);
void Call_GPU_Init(int tid,float **d_u);

void Call_Laplace(int tid, float **d_u,float **d_un);
void Call_Update(int tid, float **h_u,float **d_un);
void Save_Results(float *h_u);
