// GPU_MAIN.CU
// The CUDA (GPU) Component for a 
// multi-GPU example using MPI 
// for 1D Heat Transfer simulation
// Prof. Matthew Smith, 2016
// NCKU ME Department
// msmith@mail.ncku.edu.tw

// Contains our GPU functions etc.
#include <stdio.h>
#include "gpu_main.h"

void CPU_Compute(int rank, float *h_a, float *h_b) {
	int i;
	size_t size = NP*sizeof(float);
	// Compute the new temperature using all ranks (except 0)
	if (rank != 0) {
		for (i = 1; i <= NP; i++) {
			// Compute the new temperature if we are not at a boundary
			h_b[i] = h_a[i] + PHI*(h_a[i-1] + h_a[i+1] - 2.0*h_a[i]);
		}
		// Update the values held by h_a
		memcpy(h_a+1, h_b+1, size);
	}
}

__global__ void GPU_Calc_Temp(float *a, float *b) {
	// This function computes the new temperature and stores it in
	// b[i]. Since we know we are only using a single block with the 
	// number of threads = NP, we don't require the if statement below.

	int i = threadIdx.x+1;	// Ignore the first B/C cell	
	
	if ((i > 0) && (i <= NP)) {
		b[i] = a[i] + PHI*(a[i-1] + a[i+1] - 2.0*a[i]);
	}
	
}

void GPU_Compute(int rank, float **d_a, float **d_b) {
	// Call our GPU kernel using a single block with NP threads per block.
	// If we were to increase our problem size, we will need to change this
	// part (and our kernel) approriately.
	int threadsperblock = NP;    		// 1 block for each MPI thread
	int blockspergrid = 1;			// (using NP threads)
	size_t size = NP*sizeof(float);
	if (rank != 0) {
		// Calculate the new Temperature
		GPU_Calc_Temp<<<blockspergrid,threadsperblock>>>(*d_a, *d_b);
		// Update the d_a variable (not the ends)
		cudaMemcpy(*d_a+1, *d_b+1, size, cudaMemcpyDeviceToDevice); 
	}
}

void GPU_Send_Ends(int rank, float **h_a, float  **d_a) {
	cudaError_t Error;
	size_t size = 1*sizeof(float);
	if (rank != 0) {
		// Do a mempy of the first and last parts (our B/C cells)
		cudaMemcpy(*h_a+1, *d_a+1, size, cudaMemcpyDeviceToHost); 
		Error = cudaMemcpy(*h_a+NP, *d_a+NP, size, cudaMemcpyDeviceToHost); 
		if (DEBUG) printf("Rank %d - CUDA Error (MemCpy Ends d_a->h_a) = %s\n", 
						   rank, cudaGetErrorString(Error));		
	}
}

void GPU_Recieve_Ends(int rank, float **h_a, float  **d_a) {
	cudaError_t Error;
	size_t size = 1*sizeof(float);
	if (rank != 0) {
		// Do a mempy of the first and last parts (our B/C cells)
		cudaMemcpy(*d_a, *h_a, size, cudaMemcpyHostToDevice); 
		Error = cudaMemcpy(*d_a+NP+1, *h_a+NP+1, size, cudaMemcpyHostToDevice); 
		if (DEBUG) printf("Rank %d - CUDA Error (MemCpy Ends h_a->d_a) = %s\n", 
						   rank, cudaGetErrorString(Error));		
	}
}

void Copy_All_To_GPU(int rank, float **h_a, float **d_a, float **d_b) {

	size_t size;
	cudaError_t Error;
	// We are going to have all slave threads copy their data to GPU
	if (rank != 0) {
		// We will copy NP+2 elements into our array on the device
		size = (NP+2)*sizeof(float);
		Error = cudaMemcpy(*d_a, *h_a, size, cudaMemcpyHostToDevice);
		if (DEBUG) printf("Rank %d - CUDA Error (MemCpy h_a->d_a) = %s\n", 
		                   rank, cudaGetErrorString(Error));		
	}
}

void Copy_All_From_GPU(int rank, float **h_a, float **d_a, float **d_b) {

	size_t size;
	cudaError_t Error;
	// We are going to have all slaves copy their data from GPU
	if (rank != 0) {
		// We will copy NP+2 elements into our array from the device
		size = (NP+2)*sizeof(float);
		Error = cudaMemcpy(*h_a, *d_a, size, cudaMemcpyDeviceToHost);
		if (DEBUG) printf("Rank %d - CUDA Error (MemCpy d_a->h_a) = %s\n", 
						   rank, cudaGetErrorString(Error));		
	}
}

void Allocate_Memory(int rank, float **h_a, float **d_a, float **d_b) {
	cudaError_t Error;
	size_t size;	
	printf("Rank %d Allocating memory...", rank);
	if (rank == 0) {
		// The host thread stores all N elements
		size = N*sizeof(float);
		*h_a = (float*)malloc(size);	// Stores our solution		
	} else {
		// Slave threads only hold NP+2 elements
		size = (NP+2)*sizeof(float);
		*h_a = (float*)malloc(size);	
	}

	// Now, allocate memory on the device
	// Sometimes, an OpenMPI thread will have access to multiple devices.
	// In this case, we need to choose the device. At this time, if each
	// MPI thread runs on its own node, it will have access to its own GPU.
	// Otherwise, as this code is written, multiple MPI threads will share
	// the same GPU device. Ideally, a single MPI thread will have access to
	// its own device - in that case, the line below needs to be modified.
	
	Error = cudaSetDevice(0);
	if (DEBUG) printf("Rank %d - CUDA Error (setDevice) = %s\n", rank, cudaGetErrorString(Error));
	
	// Now to allocate the memory we will need
	if (rank == 0) {
		// The main thread doesn't perform GPU computation.
	} else {
		// We only require NP+2 elements for these variables.
		size = (NP+2)*sizeof(float);
		Error = cudaMalloc((void**)d_a, size);
		if (DEBUG) printf("Rank %d - CUDA Error (cudaMalloc d_a) = %s\n", 
						   rank, cudaGetErrorString(Error));
		Error = cudaMalloc((void**)d_b, size);
		if (DEBUG) printf("Rank %d - CUDA Error (cudaMalloc d_b) = %s\n", 
		                   rank, cudaGetErrorString(Error));
	}

	printf("Done\n");
}

void Free_Memory(int rank, float **h_a, float **d_a, float **d_b) {

	cudaError_t Error;

	// Free h_a (on the CPU)
	if (*h_a) free(*h_a);	

	// Free GPU variables - rank 0 won't participate in this activity.
	if (rank != 0) Error = cudaFree(*d_a);
	if (DEBUG) printf("Rank %d - CUDA Error (cudaFree d_a) = %s\n", 
					   rank, cudaGetErrorString(Error));
	if (rank != 0) Error = cudaFree(*d_b);
	if (DEBUG) printf("Rank %d - CUDA Error (cudaFree d_b) = %s\n", 
					   rank, cudaGetErrorString(Error));
}





