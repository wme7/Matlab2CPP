
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define DEBUG 1				// Display all error messages
#define N 1000        		// Size of problem (total number of cells)
#define N_GPU 502 			// Size of problem in each GPU (N/2 + 2, for 2 GPU's)
#define R 1.0				// Specific Gas Constant
#define GAMMA 1.4			// Ratio of Specific Heats
#define CV (R/(GAMMA-1.0))	// CV
#define DX (1.0/N)			// DX (for tube length L = 1)
#define DT (0.1*DX)			// Our time step
#define NO_STEPS 2000		// No. of time steps (total time = DT*NO_STEPS)

// Declare functions
void Manage_Memory(int phase, int tid, float **h_p, float **h_pl, float **d_p, float **d_u, float **d_Fp, float **d_Fm);
void Manage_Comms(int phase, int tid, float **h_p, float **h_pl, float **d_p, float **d_u, float **d_Fp, float **d_Fm);
void Manage_Bounds(int phase, int tid, float *h_p, float *h_pl);
void Save_Results(float *h_p);

void Call_GPU_Init(float **d_p, float **d_u, int tid);
void Call_GPU_Calc_Flux(float **d_p, float **d_Fp, float **d_Fm, int tid);
void Call_GPU_Calc_State(float **d_p, float **d_u, float **d_Fp, float **d_Fm, int tid);
