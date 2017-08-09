#include "cudamatrix_types.cuh"
#include <curand_kernel.h>
#include "curand.h"

texture<float,cudaTextureType2DLayered,cudaReadModeElementType> texref;

__global__
void fill_kernel(cudaMatrixf data,int3 dims,cudaMatrixT<curandState>random_states)
{

	unsigned int idx = threadIdx.x;
	unsigned int idy = threadIdx.y;
	unsigned int idz = threadIdx.z;

	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int gidy = blockIdx.y*blockDim.y+idy;
	unsigned int gidz  = blockIdx.z*blockDim.z+idz;

	if((gidx < dims.x)&&(gidy < dims.y)&&(gidz < dims.z))
	{
		data(gidx,gidy,gidz) = (curand_uniform(&random_states(gidx,gidy,gidz))*100);
	}

}

__global__
void setup_kernel(cudaMatrixT<curandState> random_states)
{

	unsigned int idx = threadIdx.x;
	unsigned int idy = threadIdx.y;
	unsigned int idz = threadIdx.z;

	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int gidy = blockIdx.y*blockDim.y+idy;
	unsigned int gidz  = blockIdx.z*blockDim.z+idz;

	curand_init(6446574,gidx+gridDim.x*(gidy+gridDim.y*gidz),0,&random_states(gidx,gidy,gidz));

}

__global__
void check_kernel(cudaMatrixf data,int3 dims)
{

	unsigned int idx = threadIdx.x;
	unsigned int idy = threadIdx.y;
	unsigned int idz = threadIdx.z;

	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int gidy = blockIdx.y*blockDim.y+idy;
	unsigned int gidz  = blockIdx.z*blockDim.z+idz;

	float mydata;
	float texdata;

	if((gidx < dims.x)&&(gidy < dims.y)&&(gidz < dims.z))
	{
		texdata = tex2DLayered(texref,gidx,gidy,gidz);
		mydata = data(gidx,gidy,gidz);

		printf(" mydata = %f, texdata = %f @ %i, %i\n",mydata,texdata,gidx,gidy);

	}


}







int main(void)
{
	int nx = 4;
	int ny = 4;
	int nz = 2;

	int3 dims;
	dims.x = nx;
	dims.y = ny;
	dims.z = nz;


	dim3 cudaGridSize(1,1,2);
	dim3 cudaBlockSize(8,8,1);

	cudaError status;

	cudaMatrixf initial_data(nx,ny,nz);
	cudaMatrixT<curandState> random_states(nx,ny,nz);

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaExtent extent;
	extent.width = nx; // Note, for cudaArrays the width field is the width in elements, not bytes
	extent.height = ny;
	extent.depth = nz;

	cudaArray *array = 0;



	status = cudaMalloc3DArray(&array,&desc,extent,cudaArrayLayered);

	if(status != cudaSuccess){fprintf(stderr, " malloc array %s\n", cudaGetErrorString(status));}

	printf("setup kernel \n");
	setup_kernel<<<cudaGridSize,cudaBlockSize>>>(random_states);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, " setup_kernel %s\n", cudaGetErrorString(status));}

	printf("Fill kernel \n");
	fill_kernel<<<cudaGridSize,cudaBlockSize>>>(initial_data,dims,random_states);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, " fill kernel %s\n", cudaGetErrorString(status));}

	cudaPitchedPtr matrixPtr = initial_data.getptr();

	cudaMemcpy3DParms params = {0};
	params.srcPtr = matrixPtr;
	params.dstArray = array;
	params.kind = cudaMemcpyDeviceToDevice;
	params.extent = extent;

	status = cudaMemcpy3D(&params);
	cudaThreadSynchronize();
	if(status != cudaSuccess){fprintf(stderr, " copy array %s\n", cudaGetErrorString(status));}



	cudaBindTextureToArray(texref,array);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, " bind array %s\n", cudaGetErrorString(status));}



	check_kernel<<<cudaGridSize,cudaBlockSize>>>(initial_data,dims);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, " check kernel %s\n", cudaGetErrorString(status));}

	return 0;






}






































