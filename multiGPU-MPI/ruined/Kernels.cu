#include "heat3d.h"

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

__constant__ REAL d_kx;
__constant__ REAL d_ky;
__constant__ REAL d_kz;

/*********************************************/
/* A method for checking error in CUDA calls */
/*********************************************/
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

/******************************/
/* Kernel for computing halos */
/******************************/
__global__ void copy_br_to_gc(REAL * __restrict uns, REAL * __restrict gc_unews, const unsigned int Nx, const unsigned int Ny, const unsigned int _Nz, const unsigned int pitch, const unsigned int gc_pitch, const unsigned int p)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int XY = pitch*(Ny+2);
	unsigned int k = _Nz + p*(1-_Nz);

	unsigned int o3d = i + j * pitch + k * XY;
	unsigned int o2d = j * gc_pitch + i;

	if( i < Nx+2 && j < Ny+2 && k < _Nz+2 )
	{
		gc_unews[o2d] = uns[o3d];
	}
}

/******************************/
/* Kernel for computing halos */
/******************************/
__global__ void copy_gc_to_br(REAL * __restrict uns, REAL * __restrict gc_unews, const unsigned int Nx, const unsigned int Ny, const unsigned int _Nz, const unsigned int pitch, const unsigned int gc_pitch, const unsigned int p)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int k = (1-p)*(_Nz+1);
	unsigned int o2d = j * gc_pitch + i;
	unsigned int o3d = i + j * pitch + k*(Ny+2)*pitch;

	if( i < Nx+2 && j < Ny+2 && k < _Nz+2 )
	{
		uns[o3d] = gc_unews[o2d];
	}
}

/*************************************/
/* Kernel for updating halos via P2P */
/*************************************/
__global__ void update_halo(REAL * __restrict u_target, REAL * __restrict u_source, const unsigned int Nx, const unsigned int Ny, const unsigned int _Nz, const unsigned int pitch, const unsigned int p)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int XY = pitch*(Ny+2);
	unsigned int ks = _Nz + (1-p)*(1-_Nz); //source k value
	unsigned int kt = (1-p)*(_Nz+1); //target k value (ghost cell k index)

	unsigned int o3ds = i + j * pitch + ks * XY;
	unsigned int o3dt = i + j * pitch + kt * XY;

	if( i < Nx+2 && j < Ny+2 && ks < _Nz+2 && kt < _Nz+2 )
	{
		u_target[o3dt] = u_source[o3ds];
	}
}

/**************************************************************/
/* Kernel for computing the 3D Heat equation async on the GPU */
/**************************************************************/
__global__ void heat3D_async(REAL * __restrict__ un, REAL * __restrict__ u, const unsigned int NX, const unsigned int Nx, const unsigned int Ny, const unsigned int _Nz, const unsigned int kstart, const unsigned int kstop, const unsigned int loop_z)
{
	register REAL center;
	register REAL above;
	register REAL below;
	unsigned int i, j, k, o, n, s, e, w, z, XY;
	XY = NX*(Ny+2);

    i = 1+threadIdx.x + blockIdx.x * blockDim.x;
    j = 1+threadIdx.y + blockIdx.y * blockDim.y;
    k = 1+ blockIdx.z * loop_z;

    k = max(kstart,k);

	o = i+NX*j+XY*k; //    n
	n = o + XY;      //    |
	s = o - XY;      // e--o--w
	e = o + 1;       //    |
	w = o - 1;       //    s

    if (i > 0 && i < Nx+1 && j > 0 && j < Ny+1)
    {
		below = u[o-XY]; center= u[o]; above = u[o+XY];

    	un[o] = center + d_kx*(u[e]-2*center+u[w]) + d_ky*(u[n]-2*center+u[s]) + d_kz*(below-2*center+above);

    	for(z = 1; z < loop_z; z++)
    	{
    		k += 1;

    		if (k < min(kstop, _Nz+1))
    		{
    			o = o+XY; below = center; center = above; above = u[o+XY];

				un[o] = center + d_kx*(u[e]-2*center+u[w]) + d_ky*(u[n]-2*center+u[s]) + d_kz*(below-2*center+above);
	  		}
    	}
    }
}

/****************************************************/
/* Kernel for computing 3D Heat equation on the GPU */
/****************************************************/
__global__ void heat3D(REAL * __restrict__ un, REAL * __restrict__ u, const unsigned int NX, const unsigned int Nx, const unsigned int Ny, const unsigned int _Nz, const unsigned int loop_z)
{
	register REAL center;
	register REAL above;
	register REAL below;
	unsigned int i, j, k, o, n, s, e, w, z, XY;
	XY = NX*(Ny+2);

    i = 1+threadIdx.x + blockIdx.x * blockDim.x;
    j = 1+threadIdx.y + blockIdx.y * blockDim.y;
    k = 1+threadIdx.z + blockIdx.z * loop_z;

	o = i+NX*j+XY*k; //    n
	n = o + XY;      //    |
	s = o - XY;      // e--o--w
	e = o + 1;       //    |
	w = o - 1;       //    s

    if (i > 0 && i < Nx+1 && j > 0 && j < Ny+1)
    {
		below = u[o-XY]; center= u[o]; above = u[o+XY];

    	un[o] = center + d_kx*(u[e]-2*center+u[w]) + d_ky*(u[n]-2*center+u[s]) + d_kz*(below-2*center+above);

    	for(z = 1; z < loop_z; z++)
    	{
    		o = o+XY; below = center; center= above; above = u[o+XY];

    		un[o] = center + d_kx*(u[e]-2*center+u[w]) + d_ky*(u[n]-2*center+u[s]) + d_kz*(below-2*center+above);
    	}
    }
}

/******************************************************************************/
/* Function that copies content from the host to the device's constant memory */
/******************************************************************************/
extern "C" void CopyToConstantMemory(const REAL kx, const REAL ky, const REAL kz)
{
	checkCuda(cudaMemcpyToSymbol(d_kx, &kx, sizeof(REAL), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(d_ky, &ky, sizeof(REAL), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(d_kz, &kz, sizeof(REAL), 0, cudaMemcpyHostToDevice));
}

extern "C" void ComputeInnerPoints(dim3 thread_blocks, dim3 threads_per_block, REAL* d_s_Unews, REAL* d_s_Uolds,
		int pitch, unsigned int Nx, unsigned int Ny, unsigned int _Nz)
{
	heat3D<<<thread_blocks, threads_per_block>>>(d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, k_loop);
}

extern "C" void ComputeInnerPointsAsync(dim3 thread_blocks, dim3 threads_per_block, cudaStream_t aStream, REAL* d_s_Unews, REAL* d_s_Uolds,
		int pitch, unsigned int Nx, unsigned int Ny, unsigned int _Nz, unsigned int kstart, unsigned int kstop)
{
	heat3D_async<<<thread_blocks, threads_per_block, 0, aStream>>>(d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, kstart, kstop, k_loop);
}

extern "C" void CopyBoundaryRegionToGhostCell(dim3 thread_blocks_halo, dim3 threads_per_block, REAL* d_s_Unews, REAL* d_right_send_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p)
{
	copy_br_to_gc<<<thread_blocks_halo, threads_per_block>>>(d_s_Unews, d_right_send_buffer, Nx, Ny, _Nz, pitch, gc_pitch, p);
}

extern "C" void CopyBoundaryRegionToGhostCellAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, REAL* d_s_Unews, REAL* d_right_send_buffer,
		unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p)
{
	copy_br_to_gc<<<thread_blocks_halo, threads_per_block, 0, aStream>>>(d_s_Unews, d_right_send_buffer, Nx, Ny, _Nz, pitch, gc_pitch, p);
}

extern "C" void CopyGhostCellToBoundaryRegion(dim3 thread_blocks_halo, dim3 threads_per_block, REAL* d_s_Unews, REAL* d_left_receive_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p)
{
	copy_gc_to_br<<<thread_blocks_halo, threads_per_block>>>(d_s_Unews, d_left_receive_buffer, Nx, Ny, _Nz, pitch, gc_pitch, p);
}

extern "C" void CopyGhostCellToBoundaryRegionAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, REAL* d_s_Unews, REAL* d_receive_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p)
{
	copy_gc_to_br<<<thread_blocks_halo, threads_per_block, 0, aStream>>>(d_s_Unews, d_receive_buffer, Nx, Ny, _Nz, pitch, gc_pitch, p);
}
