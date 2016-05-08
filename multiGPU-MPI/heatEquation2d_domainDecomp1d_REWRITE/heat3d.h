#ifndef _HEAT3D_H__
#define _HEAT3D_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Testing :
// A grid of n subgrids
  /* bottom
  +-------+ 
  | 0 (0) | mpi_rank (gpu)
  +-------+
  | 1 (1) |
  +-------+
     ...
  +-------+
  | n (n) |
  +-------+
    top */

/*************/
/* Constants */
/*************/
#define DEBUG 
#define WRITE 0
#define k_loop 16
#define FLOPS 8.0
#define swap(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)
#define SINE_DISTRIBUTION(i, j, k, dx, dy, dz) sin(M_PI*i*dx)*sin(M_PI*j*dy)*sin(M_PI*k*dz)
#define ROOT 0 // define root process
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { printf("MPI error calling \""#call"\"\n"); exit(-1); }

/* use floats of dobles */
#define USE_FLOAT false // set false to use real
#if USE_FLOAT
	#define REAL	float
	#define MPI_CUSTOM_REAL MPI_FLOAT
#else
	#define REAL	double
	#define MPI_CUSTOM_REAL MPI_DOUBLE
#endif

/* enviroment variable */
#define USE_OMPI true // set false for MVAPICH2
#if USE_FLOAT
	#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"
#else
	#define ENV_LOCAL_RANK "MV2_COMM_WORLD_LOCAL_RANK"
#endif

/******************/
/* Host functions */
/******************/
void InitializeMPI(int* argc, char*** argv, int* rank, int* numberOfProcesses);
void Finalize();

void init(REAL *u, const REAL dx, const REAL dy, const REAL dz, unsigned int nx, unsigned int ny, unsigned int nz);
void init_subdomain(REAL *h_s_u, REAL *h_u, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int rank);
void merge_domains(REAL *h_s_u, REAL *h_u, unsigned int nx, unsigned int ny, unsigned int _nz, unsigned int rank);
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny, unsigned int nz);
void PrintSummary(const char* kernelName, const char* optimization, double outputTimeInSeconds, double cpuTimeInSeconds, double hostToDeviceTimeInSeconds, double deviceToHostTimeInSeconds, float gflops, const int computeIterations, const int nx, const int ny, const int nz);
void CalcError(REAL *u, const REAL t, const REAL dx, const REAL dy, const REAL dz, unsigned int nx, unsigned int ny, unsigned int nz);
void Save3D(REAL *u, const unsigned int nx, const unsigned int ny, const unsigned int nz);
void Save2D(REAL *u, const unsigned int nx, const unsigned int ny);
void Save1D(REAL *u, const unsigned int nx);
void print3D(REAL *u, const unsigned int nx, const unsigned int ny, const unsigned int nz);
void print2D(REAL *u, const unsigned int nx, const unsigned int ny);
void print1D(REAL *u, const unsigned int nx);

/*******************/
/* Device wrappers */
/*******************/
extern "C"
{
	int DeviceScan();
	void AssignDevices(int rank);
	void ECCCheck(int rank);
	void CopyToConstantMemory(const REAL c0, const REAL c1, const REAL c2);
	int getBlock(int n, int block);
	void ComputeInnerPoints(dim3 thread_blocks, dim3 threads_per_block, REAL* d_s_Unews, REAL* d_s_Uolds, int pitch, unsigned int Nx, unsigned int Ny, unsigned int _Nz);
	void ComputeInnerPointsAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, REAL* d_s_Unews, REAL* d_s_Uolds,
			int pitch, unsigned int Nx, unsigned int Ny, unsigned int _Nz, unsigned int kstart, unsigned int kstop);
	void CopyBoundaryRegionToGhostCell(dim3 thread_blocks_halo, dim3 threads_per_block, REAL* d_s_Unew, REAL* d_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p);
	void CopyBoundaryRegionToGhostCellAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, REAL* d_s_Unews, REAL* d_right_send_buffer,
			unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p);
	void CopyGhostCellToBoundaryRegion(dim3 thread_blocks_halo, dim3 threads_per_block, REAL* d_s_Unew, REAL* d_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p);
	void CopyGhostCellToBoundaryRegionAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, REAL* d_s_Unews, REAL* d_receive_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p);
}

#endif	// _HEAT3D_H__
