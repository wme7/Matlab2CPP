#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define DEBUG 1 // print error messages

int main(int argc, char *argv[]) {

  int device; // counter 
  int deviceCount; // total devices

  int driverVersion;
  int runtimeVersion;

  cudaDeviceProp deviceProp;
  cudaError_t Error;

  printf("Cuda Device Query and Bandwith test \n\n");

  // Get number of devices 
  Error = cudaGetDeviceCount(&deviceCount);
  if (DEBUG) printf("CUDA error in (cudaGetDeviceCount): %s\n\n",cudaGetErrorString(Error));

  // If success (Error=0) print the number of devices and info. 
  if (Error==0) {
    printf("%d GPU found in current host.\n",deviceCount);
    
    for (device = 0; device < deviceCount; device++) {
      // Get CUDA device
      cudaSetDevice(device);
      cudaGetDeviceProperties(&deviceProp,device);
      printf("\n Device %d: \"%s\"\n",device,deviceProp.name);
      // Driver and Runtime versions
      cudaDriverGetVersion(&driverVersion);
      cudaRuntimeGetVersion(&runtimeVersion);
      printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
      printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
      // Physical mount of Global Memory and processors
      char msg[256];
      sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
	      (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
      printf("%s", msg);
      //printf("  (%2d) Multiprocessors x (%3d) CUDA Cores/MP:    %d CUDA Cores\n",
      //     deviceProp.multiProcessorCount,
      //     _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
      //     _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
      printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
      // Textures Dimensions
      printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n", 
	     deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
	     deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
      printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
	     deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
	     deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
      printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
      printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
      printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
      printf("  Warp size:                                     %d\n", deviceProp.warpSize);
      printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
      printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
      printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
      printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", deviceProp.maxGridSize[0],   deviceProp.maxGridSize[1],   deviceProp.maxGridSize[2]);
      printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
      printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
      printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
      printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
      printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
      printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
      printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
      printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
      printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
      printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", deviceProp.pciBusID, deviceProp.pciDeviceID);
      printf("  Device support overlaps from streams:          %s\n", deviceProp.deviceOverlap ? "Yes" : "No" );
      const char *sComputeMode[] =
        {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };
      printf("  Compute Mode:\n");
      printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);

    }
    // Can enable Peer-to-Peer Memory Access? 
    int P2P;
      printf("  Can Devices Access P2P: %s\n", cudaDeviceCanAccessPeer(&P2P,0,1) ? "yes" : "no");
  }
  // Otherwise quit.
  else {
    printf("cudaGetDeviceCount returned Error signal.\n");
    exit(EXIT_FAILURE);
  }
  
  return 0;
}

