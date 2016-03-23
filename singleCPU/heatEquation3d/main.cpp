
#include "heat3d.h"
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
  /*Manage_Comms(0,&h_u,&d_u);*/

  // Request computer current time
  time_t t = clock();

  //if (USE_CPU==1) {
    // Solve with CPU
    printf("Using CPU solver\n");
    for (int step=0; step < NO_STEPS; step+=2) {
      if (step%10000==0) printf("Step %d of %d\n",step,(int)NO_STEPS);
      // Compute Laplace stencil
      Call_CPU_Laplace(&h_u,&h_un); // 1st iter 
      Call_CPU_Laplace(&h_un,&h_u); // 2nd iter
    }
  /*} else {
    // Solve with GPU
    printf("Using GPU solver: %d\n",USE_GPU);
    for (int step=0; step < NO_STEPS; step+=2) {
      if (step%10000==0) printf("Step %d of %d\n",step,(int)NO_STEPS);
      // Compute Laplace stencil
      Call_GPU_Laplace(&d_u,&d_un); // 1st iter
      Call_GPU_Laplace(&d_un,&d_u); // 2nd iter
    }
    // Copy domain from device -> host
    Manage_Comms(1,&h_u,&d_u);
  }*/

  // Measure and Report computation time
  t = clock()-t; printf("Computing time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // Write solution to file
  Save_Results(h_u); 

  // Free memory on host and device
  Manage_Memory(2,&h_u,&h_un,&d_u,&d_un);

  return 0;
}
