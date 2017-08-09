/* compile as: nvcc bindingTexture2D.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024
texture<float, 2, cudaReadModeElementType> tex;
texture<float, 2, cudaReadModeElementType> tex_old;

// texture reference name must be known at compile time
__global__ void kernel() {
  int i = blockIdx.x *blockDim.x + threadIdx.x;
  int j = blockIdx.y *blockDim.y + threadIdx.y;

  float x = tex2D(tex,i,j);
  // do some work using x...
  float y = tex2D(tex_old,i,j);
  // do some work using y...
}

void call_kernel(float *buffer) {
  // bind texture to buffer
  cudaBindTexture(0, tex, buffer, N*sizeof(float));
  cudaBindTexture(0, tex_old, buffer, N*sizeof(float));

  dim3 block(128,1,1);
  dim3 grid(N/block.x,1,1);
  kernel <<<grid, block>>>();

  // unbind texture from buffer
  cudaUnbindTexture(tex);
  cudaUnbindTexture(tex_old);
}

int main() {
  // declare and allocate memory
  float *buffer;
  cudaMalloc(&buffer, N*sizeof(float));
  call_kernel(buffer);
  cudaFree(buffer);
}
