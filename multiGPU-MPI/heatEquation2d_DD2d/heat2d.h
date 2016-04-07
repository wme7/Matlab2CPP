
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define DEBUG 0 // Display all error messages
#define NX 60 // number of cells in the x-direction
#define NY 60 // number of cells in the y-direction
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
#define R 1 // radius or width of the hallo region
#define ROOT 0 // define root process
#define PI 3.1415926535897932f // PI number

/* Neighbours convention */
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

/* debuging */
#define STATUS_OK   0
#define STATUS_ERR -1

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
  int nx; // local number of cells in the x-direction 
  int ny; // local number of cells in the y-direction
  int size; // local domain size
} dmn;

/* Declare functions */
dmn Manage_Domain(int rank, int npcs, int *dim);
void Manage_Devices(int rank);
void Manage_Comms(dmn domain, double **h_u);
void Manage_Memory(int phase, dmn domain, double **g_u, double **h_u, double **h_un);
void Call_Laplace(dmn domain, double **h_u, double **h_un);
void Print_SubDomain(dmn domain, double *h_u);
void Call_IC(int IC, double *h_u);
void Save_Results(double *h_u);

