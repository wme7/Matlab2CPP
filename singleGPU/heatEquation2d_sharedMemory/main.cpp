
#include "heat2d.h"
#include <time.h>

int main() {
  // Initialize variables
  float *h_u;
  float *h_un;
  float *d_u;
  float *d_un;

  // Allocate memory in host 
  Manage_Memory(0,&h_u,&h_un,&d_u,&d_un);

  // Set Domain Initial Condition and BCs
  Call_Init(&h_u);

  // Allocate memory in device
  Manage_Memory(1,&h_u,&h_un,&d_u,&d_un);

  // Copy domain from host -> device
  Manage_Comms(0,&h_u,&d_u);

  // Request computer current time
  time_t t = clock();

  // Solver Loop (version 2)
  for (int step=0; step < NO_STEPS; step+=2) {
    if (step%100==0) printf("Step %d of %d\n",step,(int)NO_STEPS);

    // Compute stencil
    Call_GPU_Laplace(&d_u,&d_un);

    // Compute stencil (again)
    Call_GPU_Laplace(&d_un,&d_u);
  }
  // Copy domain from device -> host
  Manage_Comms(1,&h_u,&d_u);

  // Measure and Report computation time
  t = clock()-t; printf("CPU time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // Write solution to file
  Save_Results(h_u); 

  // Free memory on host and device
  Manage_Memory(2,&h_u,&h_un,&d_u,&d_un);

  return 0;
}
