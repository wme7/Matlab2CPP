
#include "heat2d.h"

void Manage_Devices(int rank) {
// Count GPUs devices and distribute them among ranks
  int devCount;
  cudaGetDeviceCount(&devCount); if (rank==ROOT) printf("Host has %d GPUs\n",devCount);
  cudaSetDevice(rank %devCount); printf("Process rank %d assigned to GPU %d\n",rank,rank%devCount);
}
