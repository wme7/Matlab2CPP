
#include "Test.h"
#include <time.h>

int main() {
  // Initialize varaibles
  float *h_u;  // global domain
  float *h_ul; // local domain
  float *d_u;  // device domain
  float *d_un; // device domain (next time step)

  // Auxiliary variables
  int tid;

  // Initialize memory only for u_l
  Manage_Memory(0,0,&h_u,&h_ul,&d_u,&d_un);
 
  // Set number of threads
  omp_set_num_threads(NO_GPU);

  // now living in a multi thread world
  #pragma omp parallel shared(h_u) private(tid,h_ul,d_u,d_un) 
  {
    // Get thread ID
    tid = omp_get_thread_num(); // tid = 0,1,2,...,NO_GPU-1
    printf("this is thread %d\n",tid);

    // Allocate u_l, d_u and d_un for each tid
    Manage_Memory(1,tid,&h_u,&h_ul,&d_u,&d_un);

    // Set Initial Condition 
    Call_GPU_Init(tid,&d_u);
    #pragma omp barrier
    
    // copy device data back to local domains
    Manage_Comms(1,tid,&h_ul,&d_u);
    
    // write local domians into global domain
    Call_Update(tid,&h_u,&h_ul);

    // Free memory
    Manage_Memory(2,tid,&h_u,&h_ul,&d_u,&d_un);
    #pragma omp barrier
  }

  // Write solution to file
  Save_Results(h_u); 

  // Free memory on host
  Manage_Memory(3,0,&h_u,&h_ul,&d_u,&d_un);

  return 0;
}
