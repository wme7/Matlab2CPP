
#include "SHLL.h"

void Manage_Memory(int phase, int tid, float **h_p, float **h_u, float **h_Fp, float **h_Fm){
	size_t size;
	if (phase == 0){
		// Allocate whole domain variables h_p on host (master thread)
		size = 3*N*sizeof(float); 	/* 3 variables per cell (rho, u, T) */
		*h_p = (float*)malloc(size);
		*h_u = (float*)malloc(size);
		*h_Fm = (float*)malloc(size);
		*h_Fp = (float*)malloc(size);	
	}
	if (phase == 1){
		// Free the whole domain variables (master thread)
		free(*h_p);
		free(*h_u);
		free(*h_Fp);
		free(*h_Fm);
	}
}

void Set_IC(float *p, float *u){

	for (int i = 0; i < N; i++)
	{
		if (i < N/2)
		{
			// Left half of shock tube
			p[3*i] = 10.0;	// Density
			p[3*i+1] = 0.0;	// Velocity
			p[3*i+2] = 1.0;	// Temperature
		}
		else if (i >= N/2)
		{	
			// Right hald of shock tube
			p[3*i] = 1.0;	// Density
			p[3*i+1] = 0.0;	// Velocity
			p[3*i+2] = 1.0;	// Temperature
		}
		// Compute Conserved quantities
		u[3*i] = p[3*i];	// Density (mass / unit volume)
		u[3*i+1] = p[3*i]*p[3*i+1]; 	// Momentum / unit volume
		u[3*i+2] = p[3*i]*(CV*p[3*i+2] + 0.5*p[3*i+1]*p[3*i+1]);	// Energy / unit volume
	}
}
	
void Call_Init(float **p, float **u){

	// Load initial condition
	Set_IC(*p, *u);
}

void Compute_Fluxes(float *p, float *Fp, float *Fm){
	float U[3], F[3], a;
	for (int i = 0; i < N; i++)
	{
		// Build flux values in every cell
		U[0] = p[3*i];			// Density
		U[1] = p[3*i]*p[3*i+1]; // Momentum
		U[2] = p[3*i]*(0.5*p[3*i+1]*p[3*i+1] + CV*p[3*i+2]); // Energy

		F[0] = p[3*i]*p[3*i+1];
		F[1] = p[3*i]*(p[3*i+1]*p[3*i+1] + R*p[3*i+2]);
		F[2] = p[3*i+1]*(U[2] + p[3*i]*R*p[3*i+2]);

		a = sqrtf(GAMMA*R*p[3*i+2]);

		// Fluxes. Pseudo-Rusanov Split Form
		Fp[3*i]   = 0.5*F[0] + a*U[0];
		Fp[3*i+1] = 0.5*F[1] + a*U[1];
		Fp[3*i+2] = 0.5*F[2] + a*U[2];

		Fm[3*i]   = 0.5*F[0] - a*U[0];
		Fm[3*i+1] = 0.5*F[1] - a*U[1];
		Fm[3*i+2] = 0.5*F[2] - a*U[2];
	}

}

void Call_Calc_Fluxes(float **p, float **Fp, float **Fm){

	// Compute fluxes
	Compute_Fluxes(*p, *Fp, *Fm);
}

void Compute_States(float *p, float *u, float *Fp, float *Fm){
	float FL[3], FR[3];
	for (int i = 1; i < N-1; i++)
	{
		// Compute net fluxes on the left and right
		// net fluxes on the left
		FL[0] = Fp[3*(i-1)]   + Fm[3*i];
		FL[1] = Fp[3*(i-1)+1] + Fm[3*i+1];
		FL[2] = Fp[3*(i-1)+2] + Fm[3*i+2];

		// net fluxes on the right
		FR[0] = Fp[3*i]   + Fm[3*(i+1)];
		FR[1] = Fp[3*i+1] + Fm[3*(i+1)+1];
		FR[2] = Fp[3*i+2] + Fm[3*(i+1)+2];

		// update the state in this cell
		u[3*i]   = u[3*i]   - (DT/DX)*(FR[0]-FL[0]);
		u[3*i+1] = u[3*i+1] - (DT/DX)*(FR[1]-FL[1]);
		u[3*i+2] = u[3*i+2] - (DT/DX)*(FR[2]-FL[2]);

		// update primitives as well
		p[3*i]   = u[3*i];
		p[3*i+1] = u[3*i+1]/u[3*i];
		p[3*i+2] = ((u[3*i+2]/u[3*i])-0.5*p[3*i+1]*p[3*i+1])/CV;
	}
}

void Call_Calc_State(float **p, float **u, float **Fp, float **Fm){

	// compute next time step states
	Compute_States(*p, *u, *Fp, *Fm);
}

void Save_Results(float *h_p){
	FILE *pFile;
	pFile = fopen("result.txt","w");
	if (pFile != NULL)
	{
		for (int i = 0; i < N; i++)
		{
			fprintf(pFile, "%d\t %g\t %g\t %g\n", i, h_p[i*3], h_p[i*3+1], h_p[i*3+2]);
		}
		fclose(pFile);
	} else {	
		printf("Unable to save to file\n");
	}
}
