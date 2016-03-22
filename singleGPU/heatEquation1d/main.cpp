
#include "heat1d.h"
#include <time.h>

int main() {
  // Initialize varaibles
  float *h_u;
  float *d_u;
  float *d_un;

  // Auxiliary variables
  int step;
  time_t t;

  // Initialize memory for h_u and h_ul
  Manage_Memory(0,&h_u,&d_u,&d_un);

  // Set Dirichlet BCs in global domain
  Call_Init(&h_u);
  
  // Allocate t_u and t_un for each tid
  Manage_Memory(1,&h_u,&d_u,&d_un);

  // Set Initial Condition
  Call_GPU_Init(&d_u);

  // Request computer current time
  t = clock();
    
  // Solver Loop 
  for (step = 0; step < NO_STEPS; step+=2) {
    if (step%100==0) printf("Step %d of %d\n",step,(int)NO_STEPS);
      
    // Compute stencil
    Call_Laplace(&d_u,&d_un);
    Call_Laplace(&d_un,&d_u);
  }

  // Copy threads data to global data variable
  Manage_Comms(2,&h_u,&d_u);

  // Free memory
  Manage_Memory(2,&h_u,&d_u,&d_un);

  // Measure and Report computation time
  t = clock()-t; printf("CPU time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // Write solution to file
  Save_Results(h_u); 

  // Free memory on host
  Manage_Memory(3,&h_u,&d_u,&d_un);

  return 0;
}
