
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEBUG 0 // Display all error messages
#define NX 10000000 // number of cells in the x-direction
#define L 1.0 // domain length
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 1.0 // tEnd, output time
#define DX (L/NX) // dx, cell size
#define DT (1/(2*C*(1/DX/DX))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define NO_STEPS 10000 //(TEND/DT) // No. of time steps
#define PI 3.1415926535897932f

/* Declare functions */
void Manage_Memory(int phase, float **h_u, float **d_u, float **d_un);
void Manage_Comms(int phase, float **h_u, float **d_un);

void Call_Init(float **h_u);
void Call_GPU_Init(float **d_u);
void Call_Laplace(float **d_u,float **d_un);
void Save_Results(float *h_u);
