/*****************************************************************************

	C-DAC Tech Workshop : hyPACK-2013
                October 15-18, 2013

  Example     : singleStream.cu

  Objective   : Write a CUDA program to add the values of two array and 
                print the execution time in ms using streams.

  Input       : None

  Output      : Execution in ms

  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

****************************************************************************/

#include<stdio.h>
#define sizeOfArray 1024*1024

/*CUDA safe call to handle the error efficiently */

void CUDA_SAFE_CALL(cudaError_t call)
{
  cudaError_t ret = call;
  //printf("RETURN FROM THE CUDA CALL:%d\t:",ret);
  switch(ret)
  {
     case cudaSuccess:
        //printf("Success\n");
        break;
    case cudaErrorInvalidValue:
     {
        printf("ERROR: InvalidValue:%i.\n",__LINE__);
        exit(-1);
        break;
     }
     case cudaErrorInvalidDevicePointer:
     {
        printf("ERROR:Invalid Device pointeri:%i.\n",__LINE__);
        exit(-1);
        break;
     }
     case cudaErrorInvalidMemcpyDirection:
     {
        printf("ERROR:Invalid memcpy direction:%i.\n",__LINE__);
        exit(-1);
        break;
     }                       
     default:
     {
        printf(" ERROR at line :%i.%d' '%s\n",__LINE__,ret,cudaGetErrorString(ret));
        exit(-1);
        break;
     }
  }
}

/*The function of this kernel is to add the values of two arrays copied from host to device*/
__global__  void arrayAddition(int *device_a, int *device_b, int *device_result)
{
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	if(threadId < sizeOfArray)
		device_result[threadId] = device_a[threadId] + device_b[threadId];
}


int main(int argc, char **argv)
{
	cudaDeviceProp prop;
	int whichDevice, *host_a, *host_b, *host_result, *device_a, *device_b, *device_result;
	
	CUDA_SAFE_CALL(cudaGetDevice(&whichDevice));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, whichDevice));
	if(!prop.deviceOverlap)
	{
		printf("Device will not handle overlaps, so no speed up from the stream \n");
		return 0;
	}
	
	cudaEvent_t start, stop;
	float elapsedTime;

	/*Cuda event created */
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));
	CUDA_SAFE_CALL(cudaEventRecord(start, 0));
	
	/*Stream created*/

	cudaStream_t stream;
	CUDA_SAFE_CALL(cudaStreamCreate(&stream));

	
	/*Allocatig memory for host array and dvice array*/
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&host_a, sizeOfArray*sizeof(int), cudaHostAllocDefault));
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&host_b, sizeOfArray*sizeof(int), cudaHostAllocDefault));
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&host_result, sizeOfArray*sizeof(int), cudaHostAllocDefault));
	
	CUDA_SAFE_CALL(cudaMalloc((void **)&device_a, sizeOfArray* sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&device_b, sizeOfArray* sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&device_result, sizeOfArray* sizeof(int)));


	
	/*Assigning values to host array */
	for(int index = 0; index < sizeOfArray; index++)
	{
		host_a[index] = rand()%10;
		host_b[index] = rand()%10;
	}
	

	/* Copying of memory from host to deice */
	CUDA_SAFE_CALL(cudaMemcpyAsync(device_a, host_a, sizeOfArray*sizeof(int), cudaMemcpyHostToDevice, stream));
	CUDA_SAFE_CALL(cudaMemcpyAsync(device_b, host_b, sizeOfArray*sizeof(int), cudaMemcpyHostToDevice, stream));
	

	/*Kernel call*/
	arrayAddition<<<sizeOfArray/256, 256, 0, stream>>>(device_a, device_b, device_result);
	CUDA_SAFE_CALL(cudaMemcpyAsync(device_result, host_result, sizeOfArray*sizeof(int), cudaMemcpyHostToDevice, stream)); 

	CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("*****************CDAC - Tech Workshop :HeGaPa2012**************\n");
	printf("   \t\t\t    july 16-22              \n\n");
	printf("Size of array : %d\n", sizeOfArray);
	printf("Time taken: %3.1f ms\n", elapsedTime);

	CUDA_SAFE_CALL(cudaFreeHost(host_a));
	CUDA_SAFE_CALL(cudaFreeHost(host_b));
	CUDA_SAFE_CALL(cudaFreeHost(host_result));

	CUDA_SAFE_CALL(cudaFree(device_a));
	CUDA_SAFE_CALL(cudaFree(device_b));
	CUDA_SAFE_CALL(cudaFree(device_result));
	
	
	return 0;
}

