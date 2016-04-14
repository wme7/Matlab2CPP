
#include "simpleMPI.h"

/***************/
/* Device code */
/***************/

// No MPI here, only CUDA
// Very simple GPU Kernel that computes square roots of input numbers
__global__ void simpleMPIKernel(float *input, float *output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = sqrt(input[tid]);
}

// Initialize an array with random data (between 0 and 1)
void initData(float *data, int dataSize) {
    for (int i = 0; i < dataSize; i++) data[i] = (float)rand() / RAND_MAX;
}

// CUDA computation on each node
void computeGPU(float *hostData, int blockSize, int gridSize) {
    int dataSize = blockSize * gridSize;

    // Allocate data on GPU memory
    float *deviceInputData = NULL;
    float *deviceOutputData = NULL;
    cudaMalloc((void **)&deviceInputData, dataSize * sizeof(float));
    cudaMalloc((void **)&deviceOutputData, dataSize * sizeof(float));

    // Copy to GPU memory
    cudaMemcpy(deviceInputData, hostData, dataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    simpleMPIKernel<<<gridSize, blockSize>>>(deviceInputData, deviceOutputData);

    // Copy data back to CPU memory
    cudaMemcpy(hostData, deviceOutputData, dataSize *sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);
}

float sum(float *data, int size) {
    float accum = 0.f;
    for (int i = 0; i < size; i++) accum += data[i];
    return accum;
}
