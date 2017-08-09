
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_Y 2

#define DIVIDE_INTO(x,y) (((x)+(y)-1)/(y)) // define No. of blocks/warps

/* Declare functions */
void Call_IC(double *d_u);
void Call_Laplace(dim3 numBlocks, dim3 threadsPerBlock, double *d_u, double *d_un);
void Save_Results(double *h_u);
void Print2D(double *u);
