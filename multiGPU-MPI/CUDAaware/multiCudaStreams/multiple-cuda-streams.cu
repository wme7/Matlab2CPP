
/************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                       October 15-18, 2013

 Example             :  multiple-cuda-streams.cu

 Objective           : Objective is to demonstrate multiple streams for    
                       addition of two vectors 
 
 Input               : None

 Output              : Time of cuda event elapsed time.                                             

 Created             : August-2013

 E-mail              : hpcfte@cdac.in     

**************************************************************************/

#include<stdio.h>

void CUDA_SAFE_CALL(cudaError_t call)
{
  cudaError_t ret = call;
  switch(ret)
  {
     case cudaSuccess:
        break;
     default:
     {
        printf(" ERROR at line :%i.%d' '%s\n",__LINE__,ret,cudaGetErrorString(ret));
        exit(-1);
        break;
     }
  }
}


#define N   (1024*1024) /* N = Vector Size */
#define FULL_DATA_SIZE   (N*20)


__global__ void kernel( int *a, int *b, int *c ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}


int main( void ) {
    cudaDeviceProp  prop;
    int whichDevice;
    CUDA_SAFE_CALL( cudaGetDevice( &whichDevice ) );
    CUDA_SAFE_CALL( cudaGetDeviceProperties( &prop, whichDevice ) );
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
        return 0;
    }
    
    if( (prop.concurrentKernels == 0 )) //check concurrent kernel support
    {
            printf("> GPU does not support concurrent kernel execution\n");
            printf("  CUDA kernel runs will be serialized\n");
    }
    if(prop.asyncEngineCount == 0) //check concurrent data transfer support
    {
            printf("GPU does not support concurrent Data transer and overlaping of kernel execution & data transfer\n");
            printf("Mem copy call will be blocking calls\n");
    }


    cudaEvent_t     start, stop;
    float           elapsedTime;

    cudaStream_t    stream0, stream1;
    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0;
    int *dev_a1, *dev_b1, *dev_c1;

    // start the timers
    CUDA_SAFE_CALL( cudaEventCreate( &start ) );
    CUDA_SAFE_CALL( cudaEventCreate( &stop ) );

    // initialize the streams
    CUDA_SAFE_CALL( cudaStreamCreate( &stream0 ) );
    CUDA_SAFE_CALL( cudaStreamCreate( &stream1 ) );

    // allocate the memory on the GPU
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_a0,
                              N * sizeof(int) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_b0,
                              N * sizeof(int) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_c0,
                              N * sizeof(int) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_a1,
                              N * sizeof(int) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_b1,
                              N * sizeof(int) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_c1,
                              N * sizeof(int) ) );

    // allocate host locked memory, used to stream
    CUDA_SAFE_CALL( cudaHostAlloc( (void**)&host_a,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );
    CUDA_SAFE_CALL( cudaHostAlloc( (void**)&host_b,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );
    CUDA_SAFE_CALL( cudaHostAlloc( (void**)&host_c,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );

    for (int i=0; i<FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
    // now loop over full data, in bite-sized chunks
    for (int i=0; i<FULL_DATA_SIZE; i+= N*2) {
        // enqueue copies of a in stream0 and stream1
        CUDA_SAFE_CALL( cudaMemcpyAsync( dev_a0, host_a+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );
        CUDA_SAFE_CALL( cudaMemcpyAsync( dev_a1, host_a+i+N,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );
        // enqueue copies of b in stream0 and stream1
        CUDA_SAFE_CALL( cudaMemcpyAsync( dev_b0, host_b+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );
        CUDA_SAFE_CALL( cudaMemcpyAsync( dev_b1, host_b+i+N,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );

        // enqueue kernels in stream0 and stream1   
        kernel<<<N/256,256,0,stream0>>>( dev_a0, dev_b0, dev_c0 );
        kernel<<<N/256,256,0,stream1>>>( dev_a1, dev_b1, dev_c1 );

        // enqueue copies of c from device to locked memory
        CUDA_SAFE_CALL( cudaMemcpyAsync( host_c+i, dev_c0,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream0 ) );
        CUDA_SAFE_CALL( cudaMemcpyAsync( host_c+i+N, dev_c1,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream1 ) );
    }
    CUDA_SAFE_CALL( cudaStreamSynchronize( stream0 ) );
    CUDA_SAFE_CALL( cudaStreamSynchronize( stream1 ) );

    CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );

  
   CUDA_SAFE_CALL( cudaEventSynchronize( stop ) );
    CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time taken:  %3.1f ms\n", elapsedTime );

    // cleanup the streams and memory
    CUDA_SAFE_CALL( cudaFreeHost( host_a ) );
    CUDA_SAFE_CALL( cudaFreeHost( host_b ) );
    CUDA_SAFE_CALL( cudaFreeHost( host_c ) );
    CUDA_SAFE_CALL( cudaFree( dev_a0 ) );
    CUDA_SAFE_CALL( cudaFree( dev_b0 ) );
    CUDA_SAFE_CALL( cudaFree( dev_c0 ) );
    CUDA_SAFE_CALL( cudaFree( dev_a1 ) );
    CUDA_SAFE_CALL( cudaFree( dev_b1 ) );
    CUDA_SAFE_CALL( cudaFree( dev_c1 ) );
    CUDA_SAFE_CALL( cudaStreamDestroy( stream0 ) );
    CUDA_SAFE_CALL( cudaStreamDestroy( stream1 ) );

    return 0;
}

