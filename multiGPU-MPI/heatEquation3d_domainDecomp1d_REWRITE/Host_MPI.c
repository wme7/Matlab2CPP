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
// Kernel for computing 3D Heat equation on the CPU
////////////////////////////////////////////////////////////////////////////////
void cpu_heat3D(REAL * __restrict__ u_new, REAL * __restrict__ u_old, const REAL c0, const REAL c1, const unsigned int max_iters, const unsigned int Nx, const unsigned int Ny, const unsigned int Nz)
{
	unsigned int j_off = (Nx+2);
	unsigned int k_off = j_off * (Ny+2);

	#pragma omp parallel default(shared)
	{
		for(unsigned int iterations = 0; iterations < max_iters; iterations++)
		{
			#pragma omp for schedule(static)
			for (unsigned int k = 1; k < (Nz+2)-1; k++)
			{
				for (unsigned int j = 1; j < (Ny+2)-1; j++)
				{
					for (unsigned int i = 1; i < (Nx+2)-1; i++)
					{
						unsigned int idx = i + j*j_off + k*k_off;
						u_new[idx] =  c1 * u_old[idx] + c0 * (u_old[idx-1] + u_old[idx+1] + u_old[idx-j_off] + u_old[idx+j_off] + u_old[idx-k_off] + u_old[idx+k_off]);
					}
				}
			}
			#pragma omp single
			{
				swap(REAL*, u_old, u_new);
			}
		}
	}
}

//////////////////////
// Initializes arrays
//////////////////////
void init(REAL *u_old, REAL *u_new, const REAL h, unsigned int Nx, unsigned int Ny, unsigned int Nz)
{
	unsigned int j_off = (Nx+2);
	unsigned int k_off = j_off * (Ny+2);

	for(unsigned int k = 0; k < (Nz+2); k++)
	{
		for (unsigned int j = 0; j < (Ny+2); j++)
		{
			for (unsigned int i = 0; i < (Nx+2); i++)
			{
				unsigned int idx = i + j*j_off + k*k_off;

		        if (i==0 || i==(Nx+2)-1 || j==0 || j==(Ny+2)-1|| k==0 || k==(Nz+2)-1)
		        {
		        	u_old[idx] = 0.; u_new [idx] = 0.;
		        }
		        else
		        {
		        	u_old[idx] = INITIAL_DISTRIBUTION(i, j, k, h);
					u_new[idx] = INITIAL_DISTRIBUTION(i, j, k, h);
		        }
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Initialize the sub-domains
////////////////////////////////////////////////////////////////////////////////
void init_subdomain(REAL *h_s_uold, REAL *h_uold, unsigned int Nx, unsigned int Ny, unsigned int _Nz, unsigned int i)
{
	int idx3d = 0, idx_sd = 0;

	for(unsigned int z = 0; z < _Nz+2; z++)
	{
		for (unsigned int y = 0; y < Ny+2; y++)
		{
			for (unsigned int x = 0; x < Nx+2; x++)
			{
				idx3d = x + y*(Nx+2) + (z+i*_Nz)*(Nx+2)*(Ny+2);
				idx_sd = x + y*(Nx+2) + z*(Nx+2)*(Ny+2);

				h_s_uold[idx_sd] = h_uold[idx3d];
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Merges the smaller sub-domains into a larger domain
////////////////////////////////////////////////////////////////////////////////
void merge_domains(REAL *h_s_Uold, REAL *h_Uold, int Nx, int Ny, int _Nz, const int i)
{
	int idx3d = 0, idx_sd = 0;

	for(int z = 1; z < _Nz+1; z++)
	{
		for (int y = 1; y < Ny+1; y++)
		{
			for (int x = 1; x < Nx+1; x++)
			{
				idx3d = x + y*(Nx+2) + (z+i*_Nz)*(Nx+2)*(Ny+2);
				idx_sd = x + y*(Nx+2) + z*(Nx+2)*(Ny+2);

				h_Uold[idx3d] = h_s_Uold[idx_sd];
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// A method that calculates the GFLOPS
////////////////////////////////////////////////////////////////////////////////
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny, unsigned int nz)
{
    return iterations*(double)((nx * ny * nz) * 1e-9 * FLOPS)/computeTimeInSeconds;
}

////////////////////////////////////////////////////////////////////////////////
// Calculates the error/L2 norm
////////////////////////////////////////////////////////////////////////////////
void CalcError(REAL *uOld, REAL *uNew, const REAL t, const REAL h, unsigned int nx, unsigned int ny, unsigned int nz)
{
	unsigned int j_off = (nx+2);
	unsigned int k_off = j_off*(ny+2);

	double error = 0., l2_uold = 0., l2_unew = 0., l2_error = 0.;

	for (unsigned int k = 1; k <= nz; k++)
	{
		for (unsigned int j = 1; j <= ny; j++)
		{
			for (unsigned int i = 1; i <= nx; i++)
			{
				unsigned int idx = i + j*j_off + k*k_off;

				REAL analytical = (exp(-3*M_PI*M_PI*t) * INITIAL_DISTRIBUTION(i, j, k, h)) - uOld[idx];
				l2_error += analytical * analytical;
				error += (uOld[idx]-uNew[idx])*(uOld[idx]-uNew[idx]);
				l2_uold += (uOld[idx])*(uOld[idx]);
				l2_unew += (uNew[idx])*(uNew[idx]);
			}
		}
	}

	l2_uold = sqrt(l2_uold/(nx*ny*nz));
	l2_unew = sqrt(l2_unew/(nx*ny*nz));
	l2_error = sqrt(l2_error*h*h*h);

	printf("RMS diff                                     :  %e\n", sqrt(error/(nx*ny*nz)));
	printf("L2 norm (GPU)                                :  %e\n", l2_uold);
	printf("L2 norm (CPU)                                :  %e\n", l2_unew);
	printf("L2 error                                     :  %e\n", l2_error);
}

////////////////////////////////////////////////////////////////////////////////
// Prints experiment summary
////////////////////////////////////////////////////////////////////////////////
void PrintSummary(const char* kernelName, const char* optimization,
		double computeTimeInSeconds, double hostToDeviceTimeInSeconds, double deviceToHostTimeInSeconds, float gflops, const int computeIterations, const int nx)
{
    printf("===========================%s=======================\n", kernelName);
    printf("Optimization                                 :  %s\n", optimization);
    printf("Kernel time ex. data transfers               :  %lf seconds\n", computeTimeInSeconds);
    printf("Data transfer(s) HtD                         :  %lf seconds\n", hostToDeviceTimeInSeconds);
    printf("Data transfer DtH                            :  %lf seconds\n", deviceToHostTimeInSeconds);
    printf("===================================================================\n");
    printf("Total effective GFLOPs                       :  %lf\n", gflops);
    printf("===================================================================\n");
    printf("3D Grid Size                                 :  %d\n", nx);
    printf("Iterations                                   :  %d\n", computeIterations);
    printf("===================================================================\n");
}

////////////////////////////////////////////////////////////////////////////////
// Prints a flattened 3D array
////////////////////////////////////////////////////////////////////////////////
void print3D(REAL *T, const unsigned int Nx, const unsigned int Ny, const unsigned int Nz)
{
	for(unsigned int z = 0; z < Nz+2; z++)
	{
		for (unsigned int y = 0; y < Ny+2; y++)
		{
			for (unsigned int x = 0; x < Nx+2; x++)
			{
				unsigned int idx3d = x + y*(Nx+2) + z*(Nx+2)*(Ny+2);
		        printf("%8.2f", T[idx3d]);
			}
		    printf("\n");
		}
		printf("\n");
	}
}

////////////////////////////////////////////////////////////////////////////////
// Prints a flattened 2D array
////////////////////////////////////////////////////////////////////////////////
void print2D(REAL *T, const unsigned int Nx, const unsigned int Ny)
{
	for (unsigned int y = 0; y < Ny+2; y++)
	{
		for (unsigned int x = 0; x < Nx+2; x++)
		{
			unsigned int idx = y * Nx+2 + x;
		    printf("%8.2f", T[idx]);
		}
		printf("\n");
	}
	printf("\n");
}

/////////////////////////////
// Function to initialize MPI
/////////////////////////////
void InitializeMPI(int* argc, char*** argv, int* rank, int* numberOfProcesses)
{
	MPI_CHECK(MPI_Init(argc, argv));
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, rank));
	MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, numberOfProcesses));
	MPI_CHECK(MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
}

/////////////////////////////
// Function to finalize MPI
/////////////////////////////
void Finalize()
{
	MPI_CHECK(MPI_Finalize());
}
