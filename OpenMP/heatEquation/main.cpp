
#include "heat2d.h"
#include <time.h>

int main()
{
  // Initialize varaibles
  time_t t;
  int step;
  int tid;
  float *h_u;
  float *t_u;

  // Initialize memory for h_u and t_u
  Manage_Memory(0,0,&h_u,&t_u);

  // Set number of threads
  omp_set_num_threads(3);
#pragma omp parallel shared(h_u) private(tid,t_u,step) 
  {

    // Get thread ID
    tid = omp_get_thread_num();

    // Allocate t_u and t_un for each thread
    Manage_Memory(1,tid,&h_u,&t_u);

    // Set Initial Condition
    Call_Init(tid,&t_u);
    #pragma omp barrier

    // Request computer current time
    t = clock();

    // Solver Loop 
    for (step = 0; step < NO_STEPS; step++) {
      if (step % 100) printf("Step %d of %d\n",step,NO_STEPS);

      // Compute stencil
    //  Call_Laplace(&u,&un);
    }

    // Copy threads data to global data variable
    Call_Update(tid,&h_u,&t_u);

    // Free memory
   //Manage_Memory(2,tid,&h_u,&t_u);
    //#pragma omp barrier
  }

  // Measure and Report computation time
  t = clock()-t; printf("CPU time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // Write solution to file
  Save_Results(h_u); 

  // Free memory on host
  //Manage_Memory(2,0,&h_u,&t_u);

  return 0;
}
