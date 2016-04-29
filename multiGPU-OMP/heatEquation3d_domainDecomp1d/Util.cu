#include "heat3d.h"

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

////////////////////////////////////////////////////////////////////////////////
// A method for checking error in CUDA calls
////////////////////////////////////////////////////////////////////////////////
inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
	#if defined(DEBUG) || defined(_DEBUG)
	if (error != cudaSuccess)
	{
		printf("checkCuda error at %s:%i: %s\n", file, line, cudaGetErrorString(cudaGetLastError()));
		exit(-1);
	}
	#endif

	return;
}

////////////////////////////////////////////////////////////////////////////////
// Checks if ECC is enabled on the devices(s)
////////////////////////////////////////////////////////////////////////////////
extern "C" void ECCCheck(int number_of_devices)
{
    cudaDeviceProp properties;
    for (int i = 0; i < number_of_devices; i++)
	{
		checkCuda(cudaSetDevice(i));
	    checkCuda(cudaGetDeviceProperties(&properties, i));

	    if (properties.ECCEnabled == 1)
	    {
	        printf("ECC is turned on for device #%d\n", i);
	    }
	    else
	    {
	        printf("ECC is turned off for device #%d\n", i);
	    }
	}
}

////////////////////////////////////////////////////////////////////////////////
// Computes the thread block size
////////////////////////////////////////////////////////////////////////////////
extern "C" int getBlock(int n, int block)
{
	return (n+2)/block + ((n+2)%block == 0?0:1);
}
