
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEBUG 1 // Display all error messages
#define NX 1024 // number of cells in the x-direction
#define XGRID 2 // No. of subdomains in the x-direction
#define NO_GPU XGRID // No. of GPUs and OMP threads
#define SNX (NX/XGRID) // subregion size

/* Declare functions */
void Manage_Memory(int phase, int tid, float **h_u, float **h_ul, float **d_u, float **d_un);
void Manage_Comms(int phase, int tid, float **h_ul, float **d_u);

void Call_GPU_Init(int tid, float **d_u);
void Call_Update(int tid, float **h_u, float **h_ul);
void Save_Results(float *h_u);
