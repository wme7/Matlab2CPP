/* ****************************************************************************
* FILE: omp_hello.c
* DESCRIPTION:
*   OpenMP Example - Hello World - C/C++ Version
*   In this simple example, the master thread forks a parallel region.
*   All threads in the team obtain their unique thread number and print it.
*   The master thread only prints the total number of threads.  Two OpenMP
*   library routines are used to obtain the number of threads and each
*   thread's number.
* AUTHOR: Manuel Diaz 2015.01.12
**************************************************************************** */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) {

int nthreads;
int tid;

	// Set number of threads 
	omp_set_num_threads(5);

	// Fork a team of threads giving them their own copies of variables 
	#pragma omp parallel private(nthreads, tid) 
	{

	  // Obtain thread number 
	  tid = omp_get_thread_num();
	  printf("This is thread number = %d\n", tid);

	  // Only master thread does this 
	  if (tid == 0) 
	    {
	    nthreads = omp_get_num_threads();
	    printf("Total number of threads = %d\n", nthreads);
	    }

	}   /* All threads join master thread and disband */ 

}
