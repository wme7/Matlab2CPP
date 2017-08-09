
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEBUG 0 // Display all error messages
#define NX 1024 // number of cells in the x-direction
#define NY 1024 // number of cells in the y-direction
#define L 10.0 // domain length
#define W 10.0 // domain width
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 1.0 // tEnd, output time
#define DX (L/NX) // dx, cell size
#define DY (W/NY) // dy, cell size
#define DT (1/(2*C*(1/DX/DX+1/DY/DY))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define KY (C*DT/(DY*DY)) // numerical conductivity
#define NO_STEPS (TEND/DT) // No. of time steps
#define PI 3.1415926535897932f
#define NI 32 // threads per block in the i-direction
#define NJ 16 // threads per block in the j-direction
#define RADIUS 1 // cells width or halo region

/* Declare functions */
void Manage_Memory(int phase, int tid, float **h_u, float **d_u, float **d_un);
void Manage_Comms(int phase, int tid, float **h_u, float **d_u);
void Call_GPU_Init(float **d_u);
void Call_Laplace(float **d_u,float **d_un);
void Call_Update(float **d_u,float **d_un);
void Save_Results(float *h_u);
