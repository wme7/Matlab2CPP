#include <mpi.h>

int main( int argc, char *argv[]){
  
  // Initialize variables
  int rank;
  float *hptr = NULL;
  float *dptr = NULL;
  const size_t elements = 32;
    
  // Initialize MPI
  MPI_Init( NULL, NULL );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // Allocate a single float in host memory
  hptr = (float*)malloc( elements*sizeof(float) );

  // Initialize array in hptr
  if (rank==0){
    for(int i=0; i<elements; i++) {hptr[i] = i; printf("%1.0f ",hptr[i]);} printf("\n");
  }

  // Allocate a single float in device memory
  cudaMalloc( (void**)&dptr, elements*sizeof(float) );

  // Send array to device
  if (rank==0) cudaMemcpy(dptr,hptr,elements*sizeof(float),cudaMemcpyHostToDevice);

  // Pass element directly from device 0 to device 1
  int tag = 0;
  MPI_Status status;
  if (rank==0) MPI_Send( dptr, elements, MPI_FLOAT, 1, tag, MPI_COMM_WORLD );
  if (rank==1) MPI_Recv( dptr, elements, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status );

  // Send array back to host
  if (rank==1) cudaMemcpy(hptr,dptr,elements*sizeof(float),cudaMemcpyDeviceToHost);

  // print the array
  if (rank==1){
    for(int i=0; i<elements; i++) printf("%1.0f ",hptr[i]);	printf("\n");
  }

  // clear memory in all devices
  free( hptr );
  cudaFree( dptr );

  // print output message
  printf("D2D -- TEST PASSED\n");

  // Finalize MPI
  MPI_Finalize();

  return 0;
}
