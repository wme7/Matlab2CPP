
#include "heat1d.h"
#include <time.h>

int main() {
  // Initialize varaibles
  float *h_u;
  float *h_un;

  // Auxiliary variables
  int step;
  time_t t;

  // Initialize memory for h_u and h_ul
  Manage_Memory(0,&h_u,&h_un);

  // Set Dirichlet BCs in global domain
  Call_Init(&h_u);
  
  // Request computer current time
  t = clock(); 
    
  // Solver Loop 
  for (step = 0; step < NO_STEPS; step+=2) {
    if (step%1000==0) printf("Step %d of %d\n",step,(int)NO_STEPS);
       
    // Compute stencil
    Call_Laplace(&h_u,&h_un); // 1st iter
    Call_Laplace(&h_un,&h_u); // 2nd iter
  }

  // Measure and Report computation time
  t = clock()-t; printf("CPU time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // Write solution to file
  Save_Results(h_u); 

  // Free memory on host
  Manage_Memory(1,&h_u,&h_un);

  return 0;
}
