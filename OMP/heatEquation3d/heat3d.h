
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEBUG 0 // Display all error messages
#define WRITE 0 // Write solution to file
#define FLOPS 8.0
#define PI 3.1415926535897932f

/* define macros */
#define I2D(n,i,j) ((i)+(n)*(j)) // transfrom a 2D array index pair into linear index memory
#define DIVIDE_INTO(x,y) (((x)+(y)-1)/(y)) // define No. of blocks/warps
#define SINE_DISTRIBUTION(i, j, k, dx, dy, dz) sin(M_PI*i*dx)*sin(M_PI*j*dy)*sin(M_PI*k*dz)
#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)

/* use floats of doubles */
#define USE_FLOAT true // set false to use double
#if USE_FLOAT
	#define REAL	float
	#define MPI_CUSTOM_REAL MPI_REAL
#else
	#define REAL	double
	#define MPI_CUSTOM_REAL MPI_DOUBLE
#endif

/* Declare functions */
void Call_Init(const int IC, REAL *h_u, const REAL dx, const REAL dy, const REAL dz, unsigned int nx, unsigned int ny, unsigned int nz);
void Call_CPU_Jacobi3d(REAL *u,REAL *un, const unsigned int max_iters, const REAL kx, const REAL ky, const REAL kz, const unsigned int nx, const unsigned int ny, const unsigned int nz);
void Call_CPU_Jacobi3d_v2(REAL *u,REAL *un, const unsigned int max_iters, const REAL kx, const REAL ky, const REAL kz, const unsigned int nx, const unsigned int ny, const unsigned int nz);
void Call_OMP_Jacobi3d(REAL *u,REAL *un, const unsigned int max_iters, const REAL kx, const REAL ky, const REAL kz, const unsigned int nx, const unsigned int ny, const unsigned int nz);
void Call_OMP_Jacobi3d_v2(REAL *u,REAL *un, const unsigned int max_iters, const REAL kx, const REAL ky, const REAL kz, const unsigned int nx, const unsigned int ny, const unsigned int nz);

float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny, unsigned int nz);
void PrintSummary(const char* kernelName, const char* optimization, REAL computeTimeInSeconds, float gflops, REAL outputTimeInSeconds,const int computeIterations, const int nx, const int ny, const int nz);
void CalcError(REAL *u, const REAL t, const REAL dx, const REAL dy, const REAL dz, unsigned int nx, unsigned int ny, unsigned int nz);

void Save3D(REAL *u, const unsigned int nx, const unsigned int ny, const unsigned int nz);
void print3D(REAL *u, const unsigned int nx, const unsigned int ny, const unsigned int nz);
