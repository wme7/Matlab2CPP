#ifndef __SYNC_H__
#define __SYNC_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*************/
/* Constants */
/*************/
#define DEBUG
#define _GC_DEPTH 1
#define k_loop 16
#define REAL double
#define FLOPS 8.0
#define swap(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)
#define INITIAL_DISTRIBUTION(i, j, k, h) sin(M_PI*i*h) * sin(M_PI*j*h) * sin(M_PI*k*h)
#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { printf("MPI error calling \""#call"\"\n"); exit(-1); }

/******************/
/* Host functions */
/******************/
void init(REAL *u_old, REAL *u_new, const REAL h, unsigned int Nx, unsigned int Ny, unsigned int Nz);
void init_subdomain(REAL *h_s_uold, REAL *h_uold, unsigned int Nx, unsigned int Ny, unsigned int Nz, unsigned int i);
void merge_domains(REAL *h_s_Uold, REAL *h_Uold, int Nx, int Ny, int _Nz, const int i);
void cpu_heat3D(REAL * __restrict__ u_new, REAL * __restrict__ u_old, const REAL c0, const REAL c1, const unsigned int max_iters, const unsigned int Nx, const unsigned int Ny, const unsigned int Nz);
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny, unsigned int nz);
void PrintSummary(const char* kernelName, const char* optimization, double computeTimeInSeconds, double hostToDeviceTimeInSeconds, double deviceToHostTimeInSeconds, float gflops, const int computeIterations, const int nx);
void CalcError(REAL *uOld, REAL *uNew, const REAL t, const REAL h, unsigned int nx, unsigned int ny, unsigned int nz);
void print3D(REAL *T, const unsigned int Nx, const unsigned int Ny, const unsigned int Nz);
void print2D(REAL *T, const unsigned int Nx, const unsigned int Ny);

/*******************/
/* Device wrappers */
/*******************/
extern "C"
{
	void CopyToConstantMemory(const REAL c0, const REAL c1);
	int getBlock(int n, int block);
	void ECCCheck(int number_of_devices);
	void ComputeInnerPoints(dim3 thread_blocks, dim3 threads_per_block, REAL* d_s_Unews, REAL* d_s_Uolds, int pitch, unsigned int Nx, unsigned int Ny, unsigned int _Nz);
	void ComputeInnerPointsAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, REAL* d_s_Unews, REAL* d_s_Uolds,
			int pitch, unsigned int Nx, unsigned int Ny, unsigned int _Nz, unsigned int kstart, unsigned int kstop);
	void CopyBoundaryRegionToGhostCell(dim3 thread_blocks_halo, dim3 threads_per_block, REAL* d_s_Unew, REAL* d_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p);
	void CopyBoundaryRegionToGhostCellAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, REAL* d_s_Unews, REAL* d_right_send_buffer,
			unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p);
	void CopyGhostCellToBoundaryRegion(dim3 thread_blocks_halo, dim3 threads_per_block, REAL* d_s_Unew, REAL* d_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p);
	void CopyGhostCellToBoundaryRegionAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, REAL* d_s_Unews, REAL* d_receive_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p);
}

#endif	// __SYNC_H__
