
#include "heat1d.h"
#include <time.h>

int main() {
  // Initialize varaibles
  float *h_u;
  float *t_u;
  float *t_un;

  // Auxiliary variables
  int tid;
  int step;
  time_t t;

  // Initialize memory for h_u 
  Manage_Memory(0,0,&h_u,&t_u,&t_un);

  // Set IC in global domain
  Call_Init(0,0,&h_u,&t_u);
  
  // Set number of threads
  omp_set_num_threads(OMP_THREADS);
  #pragma omp parallel shared(h_u) private(tid,t_u,t_un,step) 
  {
    // Get thread ID
    tid = omp_get_thread_num();

    // Allocate t_u and t_un for each tid
    Manage_Memory(1,tid,&h_u,&t_u,&t_un);

    // Set Initial Condition
    Call_Init(1,tid,&h_u,&t_u);
    #pragma omp barrier

    // Request computer current time
    t = clock(); 
    
    // Solver Loop 
    for (step = 0; step < NO_STEPS; step+=2) {
      if (step%100==0) printf("Step %d of %d\n",step,(int)NO_STEPS);
      // Communicate Boundaries
      Manage_Comms(tid,&h_u,&t_u);
      
      // Compute stencil
      Call_Laplace(tid,&t_u,&t_un);
      #pragma omp barrier

      // Communicate Boundaries (again)
      Manage_Comms(tid,&h_u,&t_un);
      
      // Compute stencil (again)
      Call_Laplace(tid,&t_un,&t_u);
    }

    // Copy threads data to global data variable
    Call_Update(tid,&h_u,&t_u);

    // Free memory
    Manage_Memory(2,tid,&h_u,&t_u,&t_un);
    #pragma omp barrier
  }

  // Measure and Report computation time
  t = clock()-t; printf("CPU time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // Write solution to file
  Save_Results(h_u); 

  // Free memory on host
  Manage_Memory(3,0,&h_u,&t_u,&t_un);

  return 0;
}
