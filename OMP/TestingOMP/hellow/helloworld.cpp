/* ****************************************************************************
* OpenMP Example - Hello World - C/C++ Version
* by Manuel Diaz 2015.01.12
**************************************************************************** */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main () {

  int nthreads;
  int tid;
  int tobj = 4.1;

  // Set number of threads 
  omp_set_num_threads(20);

  // Fork a team of threads giving them their own copies of variables 
#pragma omp parallel private(nthreads, tid)
  {
    // Obtain thread number 
    tid = omp_get_thread_num();
    printf("This is thread number = %d\n", tid);

    // Only selected thread does this 
    if ((int)tid == (int)tobj) {
      nthreads = omp_get_num_threads();
      printf("Total number of threads = %d\n", nthreads);
    }
  }/* All threads join master thread and disband */ 
}
