// Classestest.cu

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "cuPrintf.cu"
#include <ctime>
#include <cstring>
#include <cuda.h>
#include "builtin_types.h"
#include "cuda.h"
#include "cutil.h"
#include "cuda_runtime.h"
#include "host_defines.h"


typedef struct {
	int xoffset;
	int yoffset;
	int zoffset;
	int xsize;
	int ysize;
	int zsize;
} cudaMatrixcpyParams;

template <typename T>
class cudaMatrixT
{
public:
	typedef T value_type;
	cudaMatrixT(int width, int height, int depth){cudaMatrix_allocate(width, height, depth); }
	cudaMatrixT(int width, int height){cudaMatrix_allocate(width, height, 1); }
	cudaMatrixT(int width){cudaMatrix_allocate(width, 1, 1); }
	__host__ __device__ cudaMatrixT(){;}

	void cudaMatrix_allocate(int x,int y,int z)
	{

		cudaExtent h_extent;
		    h_extent.width=x*sizeof(T);
		    h_extent.height=y;
		    h_extent.depth=z;

		cudaExtent* d_extent;

		cudaMalloc((void**)&d_extent,sizeof(cudaExtent));
		cudaMemcpy(d_extent,&h_extent,sizeof(cudaExtent),cudaMemcpyHostToDevice);

		cudaPitchedPtr h_mem_device;
		cudaPitchedPtr* d_mem_device;
		cudaMalloc((void**)&d_mem_device,sizeof(cudaPitchedPtr));

		cudaMalloc3D(&h_mem_device,h_extent);

		cudaMemset3D(h_mem_device,0,h_extent);

		cudaMemcpy(d_mem_device,&h_mem_device,sizeof(cudaPitchedPtr),cudaMemcpyHostToDevice);

		extent = d_extent;
		devPitchedPtr = d_mem_device;

	}

	void cudaMatrixFree()
	{
		cudaPitchedPtr h_mem_device;
		cudaMemcpy(&h_mem_device,devPitchedPtr,sizeof(cudaPitchedPtr),cudaMemcpyDeviceToHost);
		//check_launch("Ptr Cpy");

		cudaFree(h_mem_device.ptr);
		//check_launch("Free Array");

	}
	// Device Methods

	__device__
	cudaExtent getdims_gpu(void)
	{
		return *extent;
	}

	__device__
	cudaPitchedPtr getptr_gpu(void)
	{
		return *devPitchedPtr;
	}

	cudaExtent getdims(void)
	{
		return *extent;
	}
	
	__host__
	cudaPitchedPtr getptr(void)
	{
		cudaPitchedPtr h_ptr;
		cudaMemcpy(&h_ptr,devPitchedPtr,sizeof(cudaPitchedPtr),cudaMemcpyDeviceToHost);
		return h_ptr;
	}

	__device__
	T  & operator()(int x,int y,int z)
	{
		int height = (*extent).height;
		void* devPtr = (*devPitchedPtr).ptr;
		int pitch = (*devPitchedPtr).pitch;
		int slicePitch = pitch*height;

		return ((T*)((char*)devPtr+z*slicePitch+y*pitch)+x)[0];
	}

	__device__
	const T  & operator()(int x,int y,int z)
	const
	{
		int height = (*extent).height;
		void* devPtr = (*devPitchedPtr).ptr;
		int pitch = (*devPitchedPtr).pitch;
		int slicePitch = pitch*height;

		return ((const T*)((char*)devPtr+z*slicePitch+y*pitch)+x)[0];
	}

	__device__
	T  & operator()(int x,int y)
	{
		void* devPtr = (*devPitchedPtr).ptr;
		int pitch = (*devPitchedPtr).pitch;

		return ((T*)((char*)devPtr+y*pitch)+x)[0];
	}

	__device__
	const T  & operator()(int x,int y)
	const
	{
		void* devPtr = (*devPitchedPtr).ptr;
		int pitch = (*devPitchedPtr).pitch;

		return ((const T*)((char*)devPtr+y*pitch)+x)[0];
	}

	__device__
	T  & operator()(int x)
	{
		void* devPtr = (*devPitchedPtr).ptr;

		return ((T*)((char*)devPtr)+x)[0];
	}

	__device__
	const T  & operator()(int x)
	const
	{
		void* devPtr = (*devPitchedPtr).ptr;

		return ((const T*)((char*)devPtr)+x)[0];
	}

	__host__
	void cudaMatrixcpy(const T* src,enum cudaMemcpyKind kind,cudaMatrixcpyParams* MatrixParams = NULL)
	{
		cudaPitchedPtr h_pitchedptr;
		cudaExtent h_extent;

		cudaMemcpy(&h_extent,extent,sizeof(cudaExtent),cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_pitchedptr,devPitchedPtr,sizeof(cudaPitchedPtr),cudaMemcpyDeviceToHost);

		cudaMemcpy3DParms p = { 0 };

		if (kind == cudaMemcpyHostToDevice)
		{
			if (MatrixParams == NULL)
			{
				int x = (h_extent).width/sizeof(const T);
				int y = (h_extent).height;
				int z = (h_extent).depth;


				p.srcPtr.ptr = ((void**)src);
				p.srcPtr.pitch = x*sizeof(const T);
				p.srcPtr.xsize = x;
				p.srcPtr.ysize = y;
				p.dstPtr.ptr = h_pitchedptr.ptr;
				p.dstPtr.pitch = h_pitchedptr.pitch;
				p.dstPtr.xsize = x;
				p.dstPtr.ysize = y;
				p.extent.width = x*sizeof(const T);
				p.extent.height = y;
				p.extent.depth = z;
				p.kind = cudaMemcpyHostToDevice;
			}
			else
			{
				int x = (*MatrixParams).xsize;
				int y = (*MatrixParams).ysize;
				int z = (*MatrixParams).zsize;
				int xold = (h_extent).width/sizeof(const T);
				int yold = (h_extent).height;
				int xoffset = (*MatrixParams).xoffset;
				int yoffset = (*MatrixParams).yoffset;
				int zoffset = (*MatrixParams).zoffset;

				int height = h_extent.height;
				void* devPtr = h_pitchedptr.ptr;
				int pitch = h_pitchedptr.pitch;
				int slicePitch = pitch*height;

				const T* offset = (const T*)((char*)devPtr+zoffset*slicePitch+yoffset*pitch)+xoffset;


				p.dstPtr.ptr = (void**)offset;
				p.dstPtr.pitch = h_pitchedptr.pitch;
				p.dstPtr.xsize = xold;
				p.dstPtr.ysize = yold;
				p.srcPtr.ptr = ((void**)src);
				p.srcPtr.pitch = x*sizeof(const T);
				p.srcPtr.xsize = x;
				p.srcPtr.ysize = y;
				p.extent.width = x*sizeof(const T);
				p.extent.height = y;
				p.extent.depth = z;
				p.kind = cudaMemcpyHostToDevice;
			}
		}
		else if (kind == cudaMemcpyDeviceToHost)
		{
			if (MatrixParams == NULL)
			{

				int x = h_extent.width/sizeof(const T);
				int y = h_extent.height;
				int z = h_extent.depth;


				p.srcPtr.ptr = h_pitchedptr.ptr;
				p.srcPtr.pitch = h_pitchedptr.pitch;
				p.srcPtr.xsize = x;
				p.srcPtr.ysize = y;
				p.dstPtr.ptr = ((void**)src);
				p.dstPtr.pitch = x*sizeof(const T);
				p.dstPtr.xsize = x;
				p.dstPtr.ysize = y;
				p.extent.width = x*sizeof(const T);
				p.extent.height = y;
				p.extent.depth = z;
				p.kind = cudaMemcpyDeviceToHost;
				//check_launch("Ptr Cpy");
			}
			else
			{
				int x = (*MatrixParams).xsize;
				int y = (*MatrixParams).ysize;
				int z = (*MatrixParams).zsize;
				int xold = (h_extent).width/sizeof(const T);
				int yold = (h_extent).height;
				int xoffset = (*MatrixParams).xoffset;
				int yoffset = (*MatrixParams).yoffset;
				int zoffset = (*MatrixParams).zoffset;

				int height = h_extent.height;
				void* devPtr = h_pitchedptr.ptr;
				int pitch = h_pitchedptr.pitch;
				int slicePitch = pitch*height;

				const T* offset = (const T*)((char*)devPtr+zoffset*slicePitch+yoffset*pitch)+xoffset;


				p.srcPtr.ptr = (void**)offset;
				p.srcPtr.pitch = h_pitchedptr.pitch;
				p.srcPtr.xsize = xold;
				p.srcPtr.ysize = yold;
				p.dstPtr.ptr = ((void**)src);
				p.dstPtr.pitch = x*sizeof(const T);
				p.dstPtr.xsize = x;
				p.dstPtr.ysize = y;
				p.extent.width = x*sizeof(const T);
				p.extent.height = y;
				p.extent.depth = z;
				p.kind = cudaMemcpyDeviceToHost;
			}
		}
		cudaError status = cudaMemcpy3D(&p);
		if(status != cudaSuccess){fprintf(stderr, "Memcpy: %s\n", cudaGetErrorString(status));}
	}

	// Legacy Copy Functions
	__host__
	void cudaMatrixcpyDeviceToHost(const T* dest,cudaMatrixcpyParams* MatrixParams = NULL)
	{
		cudaPitchedPtr h_pitchedptr;
		cudaExtent h_extent;

		cudaMemcpy(&h_extent,extent,sizeof(cudaExtent),cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_pitchedptr,devPitchedPtr,sizeof(cudaPitchedPtr),cudaMemcpyDeviceToHost);

		cudaMemcpy3DParms p = { 0 };

		if (MatrixParams == NULL)
		{

			int x = h_extent.width/sizeof(const T);
			int y = h_extent.height;
			int z = h_extent.depth;


			p.srcPtr.ptr = h_pitchedptr.ptr;
			p.srcPtr.pitch = h_pitchedptr.pitch;
			p.srcPtr.xsize = x;
			p.srcPtr.ysize = y;
			p.dstPtr.ptr = ((void**)dest);
			p.dstPtr.pitch = x*sizeof(const T);
			p.dstPtr.xsize = x;
			p.dstPtr.ysize = y;
			p.extent.width = x*sizeof(const T);
			p.extent.height = y;
			p.extent.depth = z;
			p.kind = cudaMemcpyDeviceToHost;
			//check_launch("Ptr Cpy");
		}
		else
		{
			int x = (*MatrixParams).xsize;
			int y = (*MatrixParams).ysize;
			int z = (*MatrixParams).zsize;
			int xold = (h_extent).width/sizeof(const T);
			int yold = (h_extent).height;
			int xoffset = (*MatrixParams).xoffset;
			int yoffset = (*MatrixParams).yoffset;
			int zoffset = (*MatrixParams).zoffset;

			int height = h_extent.height;
			void* devPtr = h_pitchedptr.ptr;
			int pitch = h_pitchedptr.pitch;
			int slicePitch = pitch*height;

			const T* offset = (const T*)((char*)devPtr+zoffset*slicePitch+yoffset*pitch)+xoffset;


			p.srcPtr.ptr = (void**)offset;
			p.srcPtr.pitch = h_pitchedptr.pitch;
			p.srcPtr.xsize = xold;
			p.srcPtr.ysize = yold;
			p.dstPtr.ptr = ((void**)dest);
			p.dstPtr.pitch = x*sizeof(const T);
			p.dstPtr.xsize = x;
			p.dstPtr.ysize = y;
			p.extent.width = x*sizeof(const T);
			p.extent.height = y;
			p.extent.depth = z;
			p.kind = cudaMemcpyDeviceToHost;
		}
		cudaError status = cudaMemcpy3D(&p);
		if(status != cudaSuccess){fprintf(stderr, "MemcpyDtH: %s\n", cudaGetErrorString(status));}
	}

	__host__
	void cudaMatrixcpyHostToDevice(const T* src,cudaMatrixcpyParams* MatrixParams = NULL)
	{
		cudaPitchedPtr h_pitchedptr;
		cudaExtent h_extent;

		cudaMemcpy(&h_extent,extent,sizeof(cudaExtent),cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_pitchedptr,devPitchedPtr,sizeof(cudaPitchedPtr),cudaMemcpyDeviceToHost);

		cudaMemcpy3DParms p = { 0 };

		if (MatrixParams == NULL)
		{
			int x = (h_extent).width/sizeof(const T);
			int y = (h_extent).height;
			int z = (h_extent).depth;


			p.srcPtr.ptr = ((void**)src);
			p.srcPtr.pitch = x*sizeof(const T);
			p.srcPtr.xsize = x;
			p.srcPtr.ysize = y;
			p.dstPtr.ptr = h_pitchedptr.ptr;
			p.dstPtr.pitch = h_pitchedptr.pitch;
			p.dstPtr.xsize = x;
			p.dstPtr.ysize = y;
			p.extent.width = x*sizeof(const T);
			p.extent.height = y;
			p.extent.depth = z;
			p.kind = cudaMemcpyHostToDevice;
		}
		else
		{
			int x = (*MatrixParams).xsize;
			int y = (*MatrixParams).ysize;
			int z = (*MatrixParams).zsize;
			int xold = (h_extent).width/sizeof(const T);
			int yold = (h_extent).height;
			int xoffset = (*MatrixParams).xoffset;
			int yoffset = (*MatrixParams).yoffset;
			int zoffset = (*MatrixParams).zoffset;

			int height = h_extent.height;
			void* devPtr = h_pitchedptr.ptr;
			int pitch = h_pitchedptr.pitch;
			int slicePitch = pitch*height;

			const T* offset = (const T*)((char*)devPtr+zoffset*slicePitch+yoffset*pitch)+xoffset;


			p.dstPtr.ptr = (void**)offset;
			p.dstPtr.pitch = h_pitchedptr.pitch;
			p.dstPtr.xsize = xold;
			p.dstPtr.ysize = yold;
			p.srcPtr.ptr = ((void**)src);
			p.srcPtr.pitch = x*sizeof(const T);
			p.srcPtr.xsize = x;
			p.srcPtr.ysize = y;
			p.extent.width = x*sizeof(const T);
			p.extent.height = y;
			p.extent.depth = z;
			p.kind = cudaMemcpyHostToDevice;
		}
		cudaError status = cudaMemcpy3D(&p);
		if(status != cudaSuccess){fprintf(stderr, "MemcpyHtD: %s\n", cudaGetErrorString(status));}
	}

	void cudaMatrixcpyFromMatrix(cudaMatrixT src,cudaMatrixcpyParams* MatrixParams = NULL)
	{
		cudaPitchedPtr h_pitchedptr;
		cudaExtent h_extent;
		cudaPitchedPtr src_pitchedptr;
		cudaExtent src_extent;

		cudaMemcpy(&h_extent,extent,sizeof(cudaExtent),cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_pitchedptr,devPitchedPtr,sizeof(cudaPitchedPtr),cudaMemcpyDeviceToHost);

		cudaMemcpy(&src_extent,src.extent,sizeof(cudaExtent),cudaMemcpyDeviceToHost);
		cudaMemcpy(&src_pitchedptr,src.devPitchedPtr,sizeof(cudaPitchedPtr),cudaMemcpyDeviceToHost);
		cudaMemcpy3DParms p = { 0 };

		if (MatrixParams == NULL)
		{
			int x = (h_extent).width/sizeof(const T);
			int y = (h_extent).height;
			int z = (h_extent).depth;


			p.srcPtr.ptr = src_pitchedptr.ptr;
			p.srcPtr.pitch = src_pitchedptr.pitch;
			p.srcPtr.xsize = x;
			p.srcPtr.ysize = y;
			p.dstPtr.ptr = h_pitchedptr.ptr;
			p.dstPtr.pitch = h_pitchedptr.pitch;
			p.dstPtr.xsize = x;
			p.dstPtr.ysize = y;
			p.extent.width = x*sizeof(const T);
			p.extent.height = y;
			p.extent.depth = z;
			p.kind = cudaMemcpyDeviceToDevice;
		}
		else
		{
			int x = (*MatrixParams).xsize;
			int y = (*MatrixParams).ysize;
			int z = (*MatrixParams).zsize;
			int xold = (h_extent).width/sizeof(const T);
			int yold = (h_extent).height;
			int xoffset = (*MatrixParams).xoffset;
			int yoffset = (*MatrixParams).yoffset;
			int zoffset = (*MatrixParams).zoffset;

			int height = h_extent.height;
			void* devPtr = h_pitchedptr.ptr;
			int pitch = h_pitchedptr.pitch;
			int slicePitch = pitch*height;

			const T* offset = (const T*)((char*)devPtr+zoffset*slicePitch+yoffset*pitch)+xoffset;


			p.dstPtr.ptr = (void**)offset;
			p.dstPtr.pitch = h_pitchedptr.pitch;
			p.dstPtr.xsize = xold;
			p.dstPtr.ysize = yold;
			p.srcPtr.ptr = src_pitchedptr.ptr;
			p.srcPtr.pitch = src_pitchedptr.pitch;
			p.srcPtr.xsize = x;
			p.srcPtr.ysize = y;
			p.extent.width = x*sizeof(const T);
			p.extent.height = y;
			p.extent.depth = z;
			p.kind = cudaMemcpyDeviceToDevice;
		}
		cudaError status = cudaMemcpy3D(&p);
		if(status != cudaSuccess){fprintf(stderr, "MemcpyDtM: %s\n", cudaGetErrorString(status));}
	}
private:
	cudaPitchedPtr* devPitchedPtr;
	cudaExtent*	extent;
};

template <typename T>
class matrix4T
{
public:
	typedef T value_type;
	__device__ __host__
    T & operator () (int row, int col) {
        return element(row,col);
    }
	__device__ __host__
    const T & operator () (int row, int col) const {
        return element(row,col);
    }
	__device__ __host__
    T & element (int row, int col) {
        return _array[row | (col<<2)];
    }
	__device__ __host__
    const T & element (int row, int col) const {
        return _array[row | (col<<2)];
    }
    // type-cast operators
	__device__ __host__
    operator T* () {
        return _array;
    }
	__device__ __host__
    operator const T* () const {
        return _array;
    }
    union {
        struct {
            T _11, _12, _13, _14;   // standard names for components
            T _21, _22, _23, _24;   // standard names for components
            T _31, _32, _33, _34;   // standard names for components
            T _41, _42, _43, _44;   // standard names for components
        };
        T _array[16];     // array access
    };
};

typedef class matrix4T<double> matrix4d;
typedef class cudaMatrixT<double> cudaMatrix;
typedef class cudaMatrixT<double> cudaMatrixd;
typedef class cudaMatrixT<double2> cudaMatrixd2;
typedef class cudaMatrixT<float> cudaMatrixf;
typedef class cudaMatrixT<int> cudaMatrixi;
typedef class cudaMatrixT<int2> cudaMatrixi2;
typedef class cudaMatrixT<int3> cudaMatrixi3;
typedef class cudaMatrixT<int4> cudaMatrixi4;
typedef class cudaMatrixT<matrix4d> cudaMatrixM4;














































