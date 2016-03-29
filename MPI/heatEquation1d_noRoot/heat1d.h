
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define DEBUG 0 // Display all error messages
#define NX 120 // number of cells in the x-direction 
#define L 1.0 // domain length
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 1.0 // tEnd, output time
#define DX (L/NX) // dx, cell size
#define DT (1/(2*C*(1/DX/DX))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define NO_STEPS (TEND/DT) // No. of time steps
#define PI 3.1415926535897932f

/* Declare functions */
int Manage_Domain(int phase, int rank, int size);
void Manage_Memory(int phase, int rank, int size, int nx, double **h_u, double **h_un);
void Manage_Comms(int rank, int size, int nx, double **h_u);
void Call_IC(int IC, int rank, int size, int nx, double *h_u);
//void Call_Laplace(int nx, float **h_u, float **h_un);
void Save_Results(int nx, double *h_u);
