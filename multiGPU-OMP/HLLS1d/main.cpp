
#include "cuda_main.h"
#include <time.h> 	// clock_t, clock, CLOCKS_PER_SEC 

int main() {
	time_t t;
	int tid;
	int step;
    // CPU (Host) variables
	float *h_p;			// Primitives Vector - entire domain
	float *h_pl;			// Primitives Vector - local (in-thread) domain
	// GPU (Device) variables
	float *d_p;			// Primitives on GPU
	float *d_u;			// Conserved quantity on GPU
	float *d_Fp;			// Forward Fluxes
	float *d_Fm;			// Backward Fluxes
	size_t size;

	// First, perform 1st phase memory management tasks
	Manage_Memory(0,0, &h_p, &h_pl, &d_p, &d_u, &d_Fp, &d_Fm);          

	// Set the number of threads
	omp_set_num_threads(2);
	#pragma omp parallel shared(h_p) private(tid, h_pl, d_p, d_u, d_Fp, d_Fm, step)
	{
		// Now living in multiple theads land. Get the Thread ID.
		tid = omp_get_thread_num();

		// Allocate memory on GPU for each thread
		Manage_Memory(1, tid, &h_p, &h_pl, &d_p, &d_u, &d_Fp, &d_Fm); 	

		// Compute the Initial Conditions on the GPU
		Call_GPU_Init(&d_p, &d_u, tid); 
		#pragma omp barrier

		// Request computers current time
		t = clock();	

		// Solver Loop
		for (step = 0; step < NO_STEPS; step++) {
			if (step % 1000 == 0) printf("Step %d of %d\n", step, NO_STEPS);

			// Synchronization work 
		    	// 1) Copy h_pl (on host) from d_p on the device
			Manage_Comms(3, tid, &h_p, &h_pl, &d_p, &d_u, &d_Fp, &d_Fm); 		
			#pragma omp barrier
			// 2) Update h_p from h_pl prior to bounds adjustment
			Manage_Bounds(-1, tid, h_p, h_pl);
			// 3) Update h_pl from h_p after bounds adjustment
			Manage_Bounds(0, tid, h_p, h_pl);
		    	// 4) Send h_pl from host to GPU (d_pl), ready for flux calculation
			Manage_Comms(2, tid, &h_p, &h_pl, &d_p, &d_u, &d_Fp, &d_Fm); 		

			// Calculate flux now
			Call_GPU_Calc_Flux(&d_p, &d_Fp, &d_Fm, tid);
			// Update state now
			Call_GPU_Calc_State(&d_p, &d_u, &d_Fp, &d_Fm, tid);
			#pragma omp barrier
		}

		// Measure computation time
		t = clock()-t;

		// Grab all results from the GPU devices and store in h_pl
		Manage_Comms(1, tid, &h_p, &h_pl, &d_p, &d_u, &d_Fp, &d_Fm); 	
		#pragma omp barrier

		// Save h_pl to h_p
		Manage_Bounds(1, tid, h_p, h_pl);
		
		// Free GPU memory on each thread
		Manage_Memory(2, tid, &h_p, &h_pl, &d_p, &d_u, &d_Fp, &d_Fm);		
		#pragma omp barrier
	}

	// Save results
	Save_Results(h_p);

	// Report time
	printf("CPU time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

	// Free last memory on host
	Manage_Memory(3,0, &h_p, &h_pl, &d_p, &d_u, &d_Fp, &d_Fm);			

	return 0;
}


