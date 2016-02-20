/* 	SHLL.h 
	The parametes of the simulation
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEBUG 0 			// Display all error messages
#define N 5000				// Number of cells in the domain
#define R 1.0				// Specific gas constant
#define GAMMA 1.4			// Ratio of specific heats
#define CV (R/(GAMMA-1.0))	// CV
#define DX (1.0/N)			// dx (size of each cell, assuming tube length L=1)
#define DT (0.1*DX)			// dt (size of the time step)
#define NO_STEPS 1000		// No. of time steps (total time = DT*NO_STEPS)

/* 	Declare functions 	*/
void Manage_Memory(int phase, int tid, float **h_p, float **h_u, float **h_Fp, float **h_Fm);

void Save_Results(float *h_p);

void Call_Init(float **h_p, float **h_u);
void Call_Calc_Fluxes(float **h_p, float **h_Fp, float **h_Fm);
void Call_Calc_State(float **h_p, float **h_u, float **h_Fp, float **h_Fm);
