
#include "SHLL.h" 	// subroutines and parameters 
#include <time.h> 	// clock_t, clock, CLOCKS_PER_SEC 

int main(){
  time_t t;
  float *h_p;	// Primitie quantities
  float *h_u;	// Conservative quantities
  float *h_Fm;	// Forward Fluxes 
  float *h_Fp;	// Backward Fluxes 

  // First, perform 1st phase memory management tasks
  Manage_Memory(0,0, &h_p, &h_u, &h_Fp, &h_Fm);

  // Compute the initial Conditions on the Host device
  Call_Init(&h_p, &h_u);

  // Request computers current time
  t = clock();

  // Solver Loop
  for (int step = 0; step < NO_STEPS; step++) {
    if(step % 100 == 0) printf("Step %d of %d \n",step,NO_STEPS);

    //calculate flux 
    Call_Calc_Fluxes(&h_p, &h_Fp, &h_Fm);
    
    //calculate state 
    Call_Calc_State(&h_p, &h_u, &h_Fp, &h_Fm);
  }

  // Measure computation time
  t = clock()-t;

  // Write solution to file
  Save_Results(h_p); /* primitives only */

  // Report time
  printf("CPU time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // Free memory on host
  Manage_Memory(1,0, &h_p, &h_u, &h_Fp, &h_Fm);
}
