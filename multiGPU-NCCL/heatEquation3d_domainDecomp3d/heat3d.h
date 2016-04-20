
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define DEBUG 0 // Display all error messages
#define NX 128 // number of cells in the x-direction
#define NY 128 // number of cells in the y-direction
#define NZ 256 // number of cells in the z-direction
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
    // A grid of 2x2x2 subgrids
    /* - layer 0 - TOP
     +-------+-------+
     |   0   |   1   |
     |(0,0,0)|(0,0,1)|
     +-------+-------+
     |   2   |   3   |
     |(0,1,0)|(0,1,1)|
     +-------+-------+
     */
    /* - layer 1 - BOTTOM
     +-------+-------+
     |   4   |   5   |
     |(1,0,0)|(1,0,1)|
     +-------+-------+
     |   6   |   7   |
     |(1,1,0)|(1,1,1)|
     +-------+-------+
     */

/* MPI Grid size */
#define SX 2 // size in x
#define SY 2 // size in y
#define SZ 2 // size in z

// neighbours convention
#define BOTTOM 0
#define TOP    1
#define NORTH  2
#define SOUTH  3
#define WEST   4
#define EAST   5

/* use floats of dobles */
#define USE_FLOAT true // set false to use double
#if USE_FLOAT
 #define real	float
 #define MPI_CUSTOM_REAL MPI_FLOAT
#else
 #define real	double
 #define MPI_CUSTOM_REAL MPI_DOUBLE
#endif

/* Declare structures */
typedef struct {
  int rank; // global rank
  int npcs; // total number of procs
  int size; // domain size (local)
  int nx; // number of cells in the x-direction (local)
  int ny; // number of cells in the y-direction (local)
  int nz; // number of cells in the z-direction (local)
  int rx; // x-rank coordinate
  int ry; // y-rank coordinate
  int rz; // z-rank coordinate
  int t; // top neigbour rank
  int b; // bottom neigbour rank
  int n; // north neigbour rank
  int s; // south neigbour rank
  int e; // east neigbour rank
  int w; // west neigbour rank
} dmn;

/* Declare functions */
dmn Manage_Domain(int rank, int npcs, int *coord, int *ngbr);
void Manage_Comms(dmn domain, MPI_Comm Comm, 
		  MPI_Datatype xySlice, MPI_Datatype yzSlice, MPI_Datatype xzSlice, real *t_u);
void Manage_Memory(int phase, dmn domain, real **h_u, real **t_u, real **t_un);
void Manage_DataTypes(int phase, dmn domain, 
		      MPI_Datatype *xySlice, MPI_Datatype *yzSlice, MPI_Datatype *xzSlice,
		      MPI_Datatype *myGlobal, MPI_Datatype *myLocal);
void Call_Laplace(dmn domain, real **t_u, real **t_un);
void Call_IC(int IC, real *h_u);
void Print(real *h_u, int nx, int ny, int nz);
void Save_Results(real *h_u);
