
#include "heat2d.h"
#include <time.h>

int main()
{
  // Initialize variables
  float *u;
  float *un;

  // Perform memory management
  Manage_Memory(0,0,&u,&un);

  // Set Initial Condition
  Call_Init(&u);

  // Request computer current time
  time_t t = clock();

  // Solver Loop (version 1)
  /*for (int step=0; step < NO_STEPS; step++) {
    if (step%100==0) printf("Step %d of %d\n",step,(int)NO_STEPS);

    // Compute Laplace stencil
    Call_Laplace(&u,&un);

    // Update solution, u=un;
    Call_Update(&u,&un); 
    }*/
  // Solver Loop (version 2)
  for (int step=0; step < NO_STEPS; step+=2) {
    if (step%100==0) printf("Step %d of %d\n",step,(int)NO_STEPS);

    // Compute Laplace stencil
    Call_Laplace(&u,&un);

    // Compute Laplace stencil (again)
    Call_Laplace(&un,&u);
  }
  // Measure and Report computation time
  t = clock()-t; printf("CPU time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // Write solution to file
  Save_Results(u); 

  // Free memory on host
  Manage_Memory(1,0,&u,&un);

  return 0;
}
