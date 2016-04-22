
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define DEBUG 0 // Display all error messages
#define NX 1024 // number of cells in the x-direction 
#define L 10.0 // domain length
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 0.1 // tEnd, output time
#define DX (L/NX) // dx, cell size
#define DT (1/(2*C*(1/DX/DX))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define NO_STEPS (TEND/DT) // No. of time steps
#define R 1 // radius of halo region
#define ROOT 0 // define root processor
#define PI 3.1415926535897932f

// Testing :
// A grid of n subgrids
  /* 
  |	   0    |    1    |     |    n    |  rank
  |---(0)---|---(1)---| ... |---(n)---|  (gpu)
  */

/* MPI Grid size */
#define SX 2 // size in x 

/* use floats of dobles */
#define USE_FLOAT true // set false to use real
#if USE_FLOAT
	#define real	float
	#define MPI_CUSTOM_REAL MPI_FLOAT
#else
	#define real	real
	#define MPI_CUSTOM_REAL MPI_real
#endif

/* Declare structures */
typedef struct {
	int gpu; // domain associated gpu
	int rank; // global rank
	int npcs; // total number of procs
	int size; // domain size (local)
	int nx; // number of cells in the x-direction (local)
	int rx; // x-rank coordinate
} dmn;

/* Declare functions */
 dmn Manage_Domain(int rank, int size, int gpu);
void Manage_Memory(int phase, dmn domain, real **h_u, real **d_u, real **d_un);
void Manage_Comms(dmn domain, real **d_u);
void Call_Laplace(dmn domain, real **d_u, real **d_un);
void Call_IC(int IC, real *h_u);
void Save_Results(real *h_u);
