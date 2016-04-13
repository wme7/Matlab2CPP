
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

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
#define R 1 // radius or width of the hallo region (fixed parameter)
#define ROOT 0 // define root process
#define PI 3.1415926535897932f // PI number

// Testing : 
    // A grid of 1x4 subgrids
    /* 
     +-----+-----+-----+-----+
     |  0  |  1  |  2  |  3  |
     |(0,0)|(0,1)|(0,2)|(0,3)|
     +-----+-----+-----+-----+
     */
    // A grid of 4x1 subgrids
    /* 
     +-----+
     |  0  |
     |(0,0)|
     +-----+
     |  1  |
     |(1,0)|
     +-----+
     |  2  |
     |(2,0)|
     +-----+
     |  3  |
     |(3,0)|
     +-----+
     */
    // A grid of 2x2 subgrids
    /* 
     +-----+-----+
     |  0  |  1  |
     |(0,0)|(0,1)|
     +-----+-----+
     |  2  |  3  |
     |(1,0)|(1,1)|
     +-----+-----+
     */

/* MPI Grid size */
#define SX 4 // size in x
#define SY 4 // size in y

/* Neighbours convention */
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

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
  int rx; // x-rank coordinate
  int ry; // y-rank coordinate
  int u; // upper neigbour rank
  int d; // lower neigbour rank
  int l; // left neigbour rank
  int r; // right neigbour rank
} dmn;

/* Declare functions */
dmn Manage_Domain(int rank, int npcs, int *coord, int *ngbr);
void Manage_Comms(dmn domain, MPI_Comm Comm, MPI_Datatype xSlice, MPI_Datatype ySlice, real *t_u);
void Manage_Memory(int phase, dmn domain, real **h_u, real **t_u, real **t_un);
void Manage_DataTypes(int phase, dmn domain, 
		      MPI_Datatype *xSlice, MPI_Datatype *ySlice, 
		      MPI_Datatype *myGlobal, MPI_Datatype *myLocal);
void Call_Laplace(dmn domain, real **t_u, real **t_un);
void Call_IC(int IC, real *h_u);
void Print(real *h_u, int nx, int ny);
void Save_Results(real *h_u);

