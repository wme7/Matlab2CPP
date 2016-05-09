#include "heat3d.h"

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

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
/* Function to initialize MPI */
/******************************/
void InitializeMPI(int *argc, char ***argv, int *rank, int *numberOfProcesses)
{
	CHECK_MPI(MPI_Init(argc, argv));
	CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, rank));
	CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, numberOfProcesses));
	CHECK_MPI(MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
}

/****************************/
/* Function to finalize MPI */
/****************************/
void Finalize()
{
	CHECK_MPI(MPI_Finalize());
}

/**********************/
/* Initializes arrays */
/**********************/
void init(REAL *u, const REAL dx, const REAL dy, const REAL dz, unsigned int nx, unsigned int ny, unsigned int nz)
{
	unsigned int i, j, k, o, xy; 
	xy = nx*ny;

	for (k = 0; k < nz; k++) {
		for (j = 0; j < ny; j++) {
			for (i = 0; i < nx; i++) {
				o = i + nx*j + xy*k;

		        if (i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1)
		        	u[o] = 0.; 
		        else
		        	u[o] = SINE_DISTRIBUTION(i, j, k, dx, dy, dz);
			}
		}
	}
}

/******************************/
/* Initialize the sub-domains */
/******************************/
void init_subdomain(REAL *h_s_u, REAL *h_u, unsigned int nx, unsigned int ny, unsigned int _nz, unsigned int rank)
{
	int idx3d = 0, idxsd = 0; unsigned int i, j, k, xy=(nx+2)*(ny+2);

	for (k = 0; k < _nz+2; k++) {
		for (j = 0; j < ny+2; j++) {
			for (i = 0; i < nx+2; i++) {
				idx3d = i + (nx+2)*j + xy*(k+rank*_nz);
				idxsd = i + (nx+2)*j + xy*k;
				h_s_u[idxsd] = h_u[idx3d];
			}
		}
	}
}

/*******************************************/
/* Merges sub-domains into a larger domain */
/*******************************************/
void merge_subdomain(REAL *h_s_u, REAL *h_u, unsigned int nx, unsigned int ny, unsigned int _nz, const int rank)
{
	int idx3d = 0, idxsd = 0; unsigned int i, j, k, xy=(nx+2)*(ny+2);

	for (k = 1; k < _nz+1; k++) {
		for (j = 1; j < ny+1; j++) {
			for (i = 1; i < nx+1; i++) {
				idx3d = i + (nx+2)*j + xy*(k+rank*_nz);
				idxsd = i + (nx+2)*j + xy*k;
				h_u[idx3d] = h_s_u[idxsd];
			}
		}
	}
}

/*******************************/
/* Prints a flattened 3D array */
/*******************************/
void print3D(REAL *u, const unsigned int nx, const unsigned int ny, const unsigned int nz)
{
	unsigned int i ,j ,k, xy;
	xy = nx*ny;
	for (k = 0; k < nz; k++)	{
		for (j = 0; j < ny; j++) {
			for (i = 0; i < nx; i++) {
				printf("%8.2f", u[i+nx*j+xy*k]);
			}
		    printf("\n");
		}
		printf("\n");
	}
}

/*******************************/
/* Prints a flattened 2D array */
/*******************************/
void print2D(REAL *u, const unsigned int nx, const unsigned int ny)
{
	unsigned int i ,j;
	for (j = 0; j < ny; j++) {
		for (i = 0; i < nx; i++) {
		    printf("%8.2f", u[i + nx*j]);
		}
		printf("\n");
	}
}

/*******************************/
/* Prints a flattened 1D array */
/*******************************/
void print1D(REAL *u, const unsigned int nx)
{
	unsigned int i;
	for (i = 0; i < nx; i++) {
	    printf("%8.2f", u[i]);
	}
	printf("\n");
}

/****************************/
/* Saves a 3D array to file */
/****************************/
void Save3D(REAL *u, const unsigned int nx, const unsigned int ny, const unsigned int nz)
{
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");  
  const int xy=nx*ny;
  if (pFile != NULL) {
    for (unsigned int k = 0;k < nz; k++) {
      	for (unsigned int j = 0; j < ny; j++) {
			for (unsigned int i = 0; i < nx; i++) {      
				fprintf(pFile, "%d\t %d\t %d\t %g\n",k,j,i,u[i+nx*j+xy*k]);
			}
      	}
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

/****************************/
/* Saves a 2D array to file */
/****************************/
void Save2D(REAL *u, const unsigned int nx, const unsigned int ny)
{
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");  
  if (pFile != NULL) {
    for (unsigned int j = 0; j < ny; j++) {
		for (unsigned int i = 0; i < nx; i++) {      
			fprintf(pFile, "%d\t %d\t %g\n",j,i,u[i+nx*j]);
		}
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

/****************************/
/* Saves a 1D array to file */
/****************************/
void Save1D(REAL *u, const unsigned int nx)
{
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");  
  if (pFile != NULL) {
	for (unsigned int i = 0; i < nx; i++) {      
		fprintf(pFile, "%d\t %g\n",i,u[i]);
	}
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

/***************************************/
/* A method that calculates the GFLOPS */
/***************************************/
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny, unsigned int nz)
{
    return iterations*(REAL)((nx*ny*nz) * 1e-9 * FLOPS)/computeTimeInSeconds;
}

/********************************/
/* Calculates the error/L2 norm */
/********************************/
void CalcError(REAL *u, const REAL t, const REAL dx, const REAL dy, const REAL dz, unsigned int nx, unsigned int ny, unsigned int nz)
{
  unsigned int i, j, k, xy;
  xy = nx*ny;

  REAL err = 0., l1_norm = 0., l2_norm = 0., linf_norm = 0.;

  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {

        err = exp(-3*M_PI*M_PI*t)*SINE_DISTRIBUTION(i,j,k,dx,dy,dz) - u[i+nx*j+xy*k];
        
        l1_norm += fabs(err);
        l2_norm += err*err;
        linf_norm = fmax(linf_norm,fabs(err));
      }
    }
  }
  
	printf("L1 norm                                      :  %e\n", dx*dy*dz*l1_norm);
	printf("L2 norm                                      :  %e\n", l2_norm);
	printf("Linf norm                                    :  %e\n", linf_norm);
}

/*****************************/
/* Prints experiment summary */
/*****************************/
void PrintSummary(const char* kernelName, const char* optimization,
		REAL computeTimeInSeconds, double outputTimeInSeconds, REAL hostToDeviceTimeInSeconds, REAL deviceToHostTimeInSeconds, 
		float gflops, const int computeIterations, const int nx, const int ny, const int nz)
{
    printf("=========================== %s =======================\n", kernelName);
    printf("Optimization                                 :  %s\n", optimization);
    printf("Kernel time ex. data transfers               :  %lf seconds\n", computeTimeInSeconds);
    printf("===================================================================\n");
    printf("Total effective GFLOPs                       :  %lf\n", gflops);
    printf("===================================================================\n");
    printf("3D Grid Size                                 :  %d x %d x %d \n", nx,ny,nz);
    printf("Iterations                                   :  %d\n", computeIterations);
    printf("Final Time                                   :  %g\n", outputTimeInSeconds);
    printf("===================================================================\n");
}
