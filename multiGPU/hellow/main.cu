
#include <stdio.h>
#include <omp.h>

int main() {
  int tid;
  int nGPU;
  cudaDeviceProp prop;

  // count number of devices
  cudaGetDeviceCount(&nGPU);

  // Set number of threads
  omp_set_num_threads(nGPU);

  // now living in multithread world
#pragma omp parallel private(prop,tid)
  {
    // get threads ids
    tid = omp_get_thread_num();

    // Print thread number
    printf("thread ID: %d\n",tid);

    // get devices properties
    cudaGetDeviceProperties(&prop, tid);
    printf("Device Number: %d\n", tid);
    printf(" Device name: %s\n", prop.name);
    printf(" Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf(" Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf(" Peak Memory Bandwidth (GB/s): %f\n\n", 
	   2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
