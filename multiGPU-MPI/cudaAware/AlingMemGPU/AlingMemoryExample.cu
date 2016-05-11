#include <stdio.h>
#include <stdlib.h>

#define N 10 	// ( N )x( N ) matrix containing data

// The idea with an aligned array is that the GPU will perform better if you pad
// it's data array so that it can fit better in cache. CUDA accomplishes this
// with the cudaMallocPitch() call. pitch (of type size_t) is the number of bytes
// per row in the array on the device. This is equivilent to
// sizeof(arraytype)*(columns + paddingColumns) OR sizeof(arraytype)*numDeviceColumns
// 
// This does mean that care must be taken when copying data to and from the device
// because the array is no longer completely linear (it has padding).
// CUDA offers a convenience function called cudaMemcpy2D() which allows you to specify
// the array rows/cols as well as the pitch for the source and the destination

__global__ void kernel(float *A, float *B, int devWidth) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int o = i+devWidth*j;

	B[o] = A[o];
	//B[o] = o; // uncomment to print the indexes
}
 
void print(float *A, int nx, int ny)
{
	for (int j = 0; j < ny; j++) {
		for (int i = 0; i < nx; i++) {
			printf("%4.0f ",A[i+nx*j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char * argv[])
{
	// Allocate on host
	float *h_A, *h_B; 
	float *d_A, *d_B;
	h_A = (float*)malloc(sizeof(float)*N*N);
	h_B = (float*)malloc(sizeof(float)*N*N);
	size_t h_pitch=sizeof(float)*N; 

	// Initialize Array h_A
 	for (int i = 0; i < N*N; i++) h_A[i]=i; //h_A[i]=0; <-- use to print the padded indexes

	// Allocate pictched memory for padded arrays d_A and d_B
	size_t d_pitch; // actual number of columns in the device array
	cudaMallocPitch((void**)&d_A,&d_pitch,sizeof(float)*N,N);
	cudaMallocPitch((void**)&d_B,&d_pitch,sizeof(float)*N,N);
	int devWidth = d_pitch/sizeof(float);
 
	// Copy memory from unpadded array A to padded array B
	cudaMemcpy2D(d_A,d_pitch,h_A,h_pitch,sizeof(float)*N,N,cudaMemcpyHostToDevice);

	dim3 threads = dim3(16,16);
	dim3 blocks = dim3((N+16-1)/16,(N+16-1)/16);
	kernel<<<blocks,threads>>>(d_A,d_B,devWidth);
	
	// Copy memory from padded array B to unpadded array A
	cudaMemcpy2D(h_B,h_pitch,d_B,d_pitch,sizeof(float)*N,N,cudaMemcpyDeviceToHost);

	// Are they the same?
 	print(h_A,N,N);
 	print(h_B,N,N);
 
 	// Release Arrays
 	cudaFree(d_A);
	cudaFree(d_B);
	free(h_A);
	free(h_B);
}
