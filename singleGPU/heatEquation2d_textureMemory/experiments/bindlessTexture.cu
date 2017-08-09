/* compile as: nvcc bindlessTexture.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024

// texture object is a kernel argument
__global__ void kernel(cudaTextureObject_t tex) {
  int i = blockIdx.x *blockDim.x + threadIdx.x;
  float x = tex1Dfetch<float>(tex, i);
  // do some work using x ...
}

void call_kernel(cudaTextureObject_t tex) {
  dim3 block(128,1,1);
  dim3 grid(N/block.x,1,1);
  kernel <<<grid, block>>>(tex);
}

int main() {
  // declare and allocate memory
  float *buffer;
  cudaMalloc(&buffer, N*sizeof(float));

  // create texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = buffer;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = 32; // bits per channel
  resDesc.res.linear.sizeInBytes = N*sizeof(float);

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  // create texture object: we only have to do this once!
  cudaTextureObject_t tex=0;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  call_kernel(tex); // pass texture as argument

  // destroy texture object
  cudaDestroyTextureObject(tex);

  cudaFree(buffer);
}
