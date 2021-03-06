
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEBUG 0 // Display all error messages
#define NX 800 // number of cells in the x-direction
#define NY 800 // number of cells in the y-direction
#define L 40.0 // domain length
#define W 20.0 // domain width
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 1.0 // tEnd, output time
#define DX (L/NX) // dx, cell size
#define DY (W/NY) // dy, cell size
#define DT (1/(2*C*(1/DX/DX+1/DY/DY))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define KY (C*DT/(DY*DY)) // numerical conductivity
#define NO_STEPS (TEND/DT) // No. of time steps
#define NO_GPU 2 // No. of GPUs and OMP threads
#define NO_threads 6 // No. GPU threads [dimBlock(NO_threads,NO_threads)]
#define XGRID NO_GPU // No. of subdomains in the x-direction
#define YGRID 1 // No. of subdomains in the y-direction
#define SNX (NX/XGRID) // subregion size
#define SNY (NY/YGRID) // subregion size
#define PI 3.1415926535897932f

/* Declare functions */
void Manage_Memory(int phase, int tid, float **h_u, float **h_ul, float **d_u, float **d_un);
void Manage_Comms(int phase, int tid, float **h_u, float **h_ul, float **d_un);

void Call_CPU_Init(float **h_u);
void Call_GPU_Init(int tid, float **d_u);
void Call_Laplace(int tid, float **d_u,float **d_un);

void Save_Results_Tid(int tid, float *h_ul);
void Save_Results(float *h_u);
