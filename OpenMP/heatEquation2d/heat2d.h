
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEBUG 1 // Display all error messages
#define NX 128 // number of cells in the x-direction
#define NY 128 // number of cells in the y-direction
#define L 1.0 // domain length
#define W 1.0 // domain width
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 1.0 // tEnd, output time
#define DX (L/NX) // dx, cell size
#define DY (W/NY) // dy, cell size
#define DT (1/(2*C*(1/DX/DX+1/DY/DY))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define KY (C*DT/(DY*DY)) // numerical conductivity
#define NO_STEPS (TEND/DT) // No. of time steps
#define XGRID 4 // No. of subdomains in the x-direction
#define YGRID 1 // No. of subdomains in the y-direction
#define OMP_THREADS XGRID*YGRID // No. of OMP threads
#define SNX (NX/XGRID) // subregion size + BC cells
#define SNY (NY/YGRID) // subregion size + BC cells
#define PI 3.1415926535897932f

/* Declare functions */
void Manage_Memory(int phase, int tid, float **h_u, float **t_u, float **t_un);
void Manage_Comms(int phase, int tid, float **t_u, float **t_un);

void Call_Init_globalDomain(float **h_u);
void Call_Init_localDomain(int tid, float **t_u);
void Call_Laplace(float **t_u,float **t_un);

void Save_Results_Tid(int tid, float *t_u);
void Save_Results(float *h_u);
