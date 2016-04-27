
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define DEBUG 1 // Display all error messages
#define NX 64 // number of cells in the x-direction
#define NY 64 // number of cells in the y-direction
#define NZ 128 // number of cells in the z-direction
#define L 10.0 // domain length
#define W 10.0 // domain width
#define H 20.0 // domain width
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 1.0 // tEnd, output time
#define DX (L/NX) // dx, cell size
#define DY (W/NY) // dy, cell size
#define DZ (H/NZ) // dy, cell size	
#define DT (1/(2*C*(1/DX/DX+1/DY/DY+1/DZ/DZ))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define KY (C*DT/(DY*DY)) // numerical conductivity
#define KZ (C*DT/(DZ*DZ)) // numerical conductivity
#define NO_STEPS (TEND/DT) // No. of time steps
#define R 1 // radius or width of the hallo region
#define ROOT 0 // define root process
#define PI 3.1415926535897932f // PI number

// Testing :
// A grid of n subgrids
  /*
  +-------+ 
  | 0 (0) | mpi_rank (gpu)
  +-------+
  | 1 (1) |
  +-------+
     ...
  +-------+
  | n (n) |
  +-------+
  */

/* MPI Grid size */
#define SX 1 // size in x <-- fix parameter!
#define SY 1 // size in y <-- fix parameter!
#define SZ 2 // size in z

/* use floats of dobles */
#define USE_FLOAT true // set false to use real
#if USE_FLOAT
	#define real	float
	#define MPI_CUSTOM_REAL MPI_FLOAT
#else
	#define real	double
	#define MPI_CUSTOM_REAL MPI_DOUBLE
#endif

/* enviroment variable */
#define USE_OMPI true // set false for MVAPICH2
#if USE_FLOAT
	#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"
#else
	#define ENV_LOCAL_RANK "MV2_COMM_WORLD_LOCAL_RANK"
#endif

/* Declare structures */
typedef struct {
	int gpu; // domain associated gpu
	int rank; // global rank
	int npcs; // total number of procs
	int size; // domain size (local)
	int nx; // local number of cells in the x-direction 
	int ny; // local number of cells in the y-direction
	int nz; // local number of cells in the z-direction
	int rx; // y-rank coordinate
	int ry; // y-rank coordinate
	int rz; // z-rank coordinate
} dmn;

/* Declare functions */
void Manage_Devices();
 dmn Manage_Domain(int rank, int npcs, int gpu);
void Manage_Memory(int phase, dmn domain, real **h_u, real **t_u, real **d_u, real **d_un);
void Manage_Comms(int phase, dmn domain, real **t_u, real **d_u);
extern "C" void Call_Laplace(dmn domain, real **h_u, real **h_un);
void Call_IC(int IC, real *h_u);
void Save_Results(real *h_u);
