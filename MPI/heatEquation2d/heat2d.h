
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define DEBUG 0 // Display all error messages
#define NX 100 // number of cells in the x-direction
#define NY 100 // number of cells in the y-direction
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
#define ROOT 0 // define root procs 
#define SX 4 // sub-grids in the x-direction
#define SY 1 // sub-grids in the y-direction
#define PI 3.1415926535897932f // PI number

/* Declare structures */
typedef struct {
  int rank; // global rank
  int npcs; // total number of procs
  int size; // local domain size
  int rx; // x-rank of local grid
  int ry; // y-rank of local grid
  int nx; // local number of cells in the x-direction 
  int ny; // local number of cells in the y-direction
} dmn;

/* Declare functions */
dmn Manage_Domain(int rank, int npcs);
void Manage_Memory(int phase, dmn domain, double **g_u, double **h_u, double **h_un);
void Manage_Comms(int phase, dmn domain, double **h_u);

void Call_Laplace(dmn domain, double **h_u, double **h_un);
void Call_IC(int IC, double *h_u);

void Save_Results(double *h_u);
