
#include "heat2d.h"
#include <time.h>

int main() {
  // Initialize varaibles
  float *h_u;
  float *d_u;
  float *d_un;

  // Auxiliary variables
  int tid;
  int step;
  time_t t;

  // Initialize memory for h_u and h_ul
  Manage_Memory(0,0,&h_u,&d_u,&d_un);

  // Set Dirichlet BCs in global domain
  Call_Init(&h_u);
  
  // Set number of threads
  omp_set_num_threads(NO_GPU);
#pragma omp parallel shared(h_u) private(tid,d_u,d_un,step) 
  {
    // Get thread ID
    tid = omp_get_thread_num(); printf("tid = %d\n",tid);

    // Allocate t_u and t_un for each tid
    Manage_Memory(1,tid,&h_u,&d_u,&d_un);

    // Set Initial Condition
    Call_GPU_Init(tid,&d_u);
    #pragma omp barrier

    // Request computer current time
    t = clock(); 
    
    // Solver Loop 
    for (step = 0; step < NO_STEPS; step+=2) {
      if (step%100==0) printf("Step %d of %d\n",step,(int)NO_STEPS);
      
      // Communicate Boundaries
      Manage_Comms(1,tid,&h_u,&d_u);
      #pragma omp barrier
      Manage_Comms(2,tid,&h_u,&d_u);
      #pragma omp barrier
    
      // Compute stencil
      Call_Laplace(tid,&d_u,&d_un);
      #pragma omp barrier

      // Communicate Boundaries
      Manage_Comms(1,tid,&h_u,&d_un);
      #pragma omp barrier
      Manage_Comms(2,tid,&h_u,&d_un);
      #pragma omp barrier
    
      // Compute stencil
      Call_Laplace(tid,&d_un,&d_u);
      #pragma omp barrier
    }

    // Copy threads data to global data variable
    Manage_Comms(3,tid,&h_u,&d_u);

    // Free memory
    Manage_Memory(2,tid,&h_u,&d_u,&d_un);
    #pragma omp barrier
  }

  // Measure and Report computation time
  t = clock()-t; printf("CPU time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // Write solution to file
  Save_Results(h_u); 

  // Free memory on host
  Manage_Memory(3,0,&h_u,&d_u,&d_un);

  return 0;
}
