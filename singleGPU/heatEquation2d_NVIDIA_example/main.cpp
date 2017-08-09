
#include "heat2d.h"
#include <time.h>

int main() {
  // Initialize variables
  float *h_u;
  float *d_u;
  float *d_un;

  // Allocate memory in host 
  Manage_Memory(0,0,&h_u,&d_u,&d_un);

  // Allocate memory in device
  Manage_Memory(1,0,&h_u,&d_u,&d_un);

  // Set Initial Condition
  Call_GPU_Init(&d_u);

  // Request computer current time
  time_t t = clock();

  // Solver Loop 
  for (int step=0; step < NO_STEPS; step+=2) {
    if (step%10000==0) printf("Step %d of %d\n",step,(int)NO_STEPS);

    // Compute stencil
    Call_Laplace(&d_u,&d_un); // 1sr iter
    Call_Laplace(&d_un,&d_u); // 2nd iter
  }
  // Copy data from device -> host
  Manage_Comms(1,0,&h_u,&d_u);

  // Measure and Report computation time
  t = clock()-t; printf("CPU time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // Write solution to file
  Save_Results(h_u); 

  // Free memory on host and device
  Manage_Memory(2,0,&h_u,&d_u,&d_un);

  return 0;
}
