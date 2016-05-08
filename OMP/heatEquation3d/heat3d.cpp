
#include "heat3d.h"
/*******************************/
/* Prints a flattened 3D array */
/*******************************/
void print3D(REAL *u, const unsigned int nx, const unsigned int ny, const unsigned int nz)
{
  unsigned int i, j, k, xy;
  xy=nx*ny;

  for(k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        printf("%8.2f", u[i+nx*j+xy*k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

/**************************/
/* Write to file 3D array */
/**************************/
void Save3D(REAL *u, const unsigned int nx, const unsigned int ny, const unsigned int nz)
{
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");  
  unsigned int xy=nx*ny;
  if (pFile != NULL) {
    for (int k = 0; k < nz; k++) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {      
          fprintf(pFile, "%d\t %d\t %d\t %g\n",k,j,i,u[i+nx*j+xy*k]);
      	}
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/

void Call_Init(const int IC, REAL *u0,
  const REAL dx, const REAL dy, const REAL dz, 
  unsigned int nx, unsigned int ny, unsigned int nz)
{
  unsigned int i, j, k, o, xy; 
  xy=nx*ny;

  switch (IC) {
    case 1: {
      for (k = 0; k < nz; k++) {
        for (j = 0; j < ny; j++) {
          for (i = 0; i < nx; i++) {
            // set all domain's cells equal to zero
            o = i+nx*j+xy*k;  u0[o] = 0.0;
            // set BCs in the domain 
            if (k==0)    u0[o] = 1.0; // bottom
            if (k==nz-1) u0[o] = 1.0; // top
          }
        }
      }
      break;
    }
    case 2: {
      for (k = 0; k < nz; k++) {
        for (j = 0; j < ny; j++) {
          for (i = 0; i < nx; i++) {
            // set all domain's cells equal to :
            o = i+nx*j+xy*k;  
            u0[o] = 1.0*exp(
              -(dx*(i-nx/2))*(dx*(i-nx/2))/1.5
              -(dy*(j-ny/2))*(dy*(j-ny/2))/1.5
              -(dz*(k-nz/2))*(dz*(k-nz/2))/12);
          }
        }
      }
      break;
    }
    case 3: {
      for (k = 0; k < nz; k++) {
        for (j = 0; j < ny; j++) {
          for (i = 0; i < nx; i++) {
            // set all domain's cells equal to :
            o = i+nx*j+xy*k;  u0[o] = SINE_DISTRIBUTION(i,j,k,dx,dy,dz); 
            // set BCs in the domain 
            if (i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) u0[o] = 0.0;
          }
        }
      }
      break;
    }
    // here to add another IC
  }
}

/************************/
/* JACOBI SOLVERS - CPU */
/************************/
void Call_CPU_Jacobi3d(REAL *u,REAL *un, const unsigned int max_iters,
  const REAL kx, const REAL ky, const REAL kz, 
  const unsigned int nx, const unsigned int ny, const unsigned int nz)
{
  // Using (i,j,k) = [i+N*j+M*N*k] indexes
  unsigned int i, j, k, o, n, s, e, w, t, b, xy;
  xy=nx*ny;

  for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
  {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        for (k = 0; k < nz; k++) {

          o = i+ (nx*j) + (xy*k); // node( j,i,k )      n  b
        	n = (i==nx-1) ? o:o+nx; // node(j+1,i,k)      | /
        	s = (i==0)    ? o:o-nx; // node(j-1,i,k)      |/
        	e = (j==ny-1) ? o:o+1;  // node(j,i+1,k)  w---o---e
        	w = (j==0)    ? o:o-1;  // node(j,i-1,k)     /|
        	t = (k==nz-1) ? o:o+xy; // node(j,i,k+1)    / |
        	b = (k==0)    ? o:o-xy; // node(j,i,k-1)   t  s

        	un[o] = u[o] + kx*(u[e]-2*u[o]+u[w]) + ky*(u[n]-2*u[o]+u[s]) + kz*(u[t]-2*u[o]+u[b]);
        }
      } 
    }
  SWAP(REAL*, u, un);
  }
}

void Call_CPU_Jacobi3d_v2(REAL *u, REAL *un, const unsigned int max_iters,
  const REAL kx, const REAL ky, const REAL kz, 
  const unsigned int nx, const unsigned int ny, const unsigned int nz)
{
  // Using (i,j,k) = [i+N*j+M*N*k] indexes
  unsigned int i, j, k, o, n, s, e, w, t, b, xy;
  xy = nx*ny;

  for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
  {
    for (k = 0; k < nz; k++) {
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {

          o = i+(nx*j)+(xy*k); // node( j,i,k )      n  b
          n = o+nx;            // node(j+1,i,k)      | /
          s = o-nx;            // node(j-1,i,k)      |/
          e = o+1;             // node(j,i+1,k)  w---o---e
          w = o-1;             // node(j,i-1,k)     /|
          t = o+xy;            // node(j,i,k+1)    / |
          b = o-xy;            // node(j,i,k-1)   t  s

          if (i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1)
            un[o] = u[o] + kx*(u[e]-2*u[o]+u[w]) + ky*(u[n]-2*u[o]+u[s]) + kz*(u[t]-2*u[o]+u[b]);
          else
            un[o] = u[o];
        }
      }
    }
  SWAP(REAL*, u, un);
  }
}

/************************/
/* JACOBI SOLVERS - OMP */
/************************/
void Call_OMP_Jacobi3d(REAL *u,REAL *un, const unsigned int max_iters,
  const REAL kx, const REAL ky, const REAL kz, 
  const unsigned int nx, const unsigned int ny, const unsigned int nz)
{
  // Using (i,j,k) = [i+N*j+M*N*k] indexes
  unsigned int i, j, k, o, n, s, e, w, t, b, xy;
  xy=nx*ny;

  #pragma omp parallel default(shared) 
  {
    for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
    {
      #pragma omp for schedule(static)
      for (k = 0; k < nz; k++) {
        for (j = 0; j < ny; j++) {
          for (i = 0; i < nx; i++) {

            o = i+ (nx*j) + (xy*k); // node( j,i,k )      n  b
            n = (i==nx-1) ? o:o+nx; // node(j+1,i,k)      | /
            s = (i==0)    ? o:o-nx; // node(j-1,i,k)      |/
            e = (j==ny-1) ? o:o+1;  // node(j,i+1,k)  w---o---e
            w = (j==0)    ? o:o-1;  // node(j,i-1,k)     /|
            t = (k==nz-1) ? o:o+xy; // node(j,i,k+1)    / |
            b = (k==0)    ? o:o-xy; // node(j,i,k-1)   t  s

            un[o] = u[o] + kx*(u[e]-2*u[o]+u[w]) + ky*(u[n]-2*u[o]+u[s]) + kz*(u[t]-2*u[o]+u[b]);
          }
        }
      }
      #pragma omp single
      {
        SWAP(REAL*, u, un);
      }
    }
  }
}

void Call_OMP_Jacobi3d_v2(REAL *u, REAL *un, const unsigned int max_iters,
  const REAL kx, const REAL ky, const REAL kz, 
  const unsigned int nx, const unsigned int ny, const unsigned int nz)
{
  // Using (i,j,k) = [i+N*j+M*N*k] indexes
	unsigned int i, j, k, o, n, s, e, w, t, b, xy;
  xy = nx*ny;

	#pragma omp parallel default(shared) 
  {
		for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
    {
			#pragma omp for schedule(static)
			for (k = 0; k < nz; k++) {
				for (j = 0; j < ny; j++) {
					for (i = 0; i < nx; i++) {

            o = i+nx*j+xy*k; // node( j,i,k )      n  b
            n = o+nx;        // node(j+1,i,k)      | /
            s = o-nx;        // node(j-1,i,k)      |/
            e = o+1;         // node(j,i+1,k)  w---o---e
            w = o-1;         // node(j,i-1,k)     /|
            t = o+xy;        // node(j,i,k+1)    / |
            b = o-xy;        // node(j,i,k-1)   t  s

            if (i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1)
              un[o] = u[o] + kx*(u[e]-2*u[o]+u[w]) + ky*(u[n]-2*u[o]+u[s]) + kz*(u[t]-2*u[o]+u[b]);
            else
              un[o] = u[o];
					}
				}
			}
			#pragma omp single
			{
				SWAP(REAL*, u, un);
			}
		}
	}
}

/******************/
/* COMPUTE GFLOPS */
/******************/
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny, unsigned int nz)
{
    return iterations*(double)((nx*ny*nz) * 1e-9 * FLOPS)/computeTimeInSeconds;
}

/***********************/
/* COMPUTE ERROR NORMS */
/***********************/
void CalcError(REAL *u, const REAL t, const REAL dx, const REAL dy, const REAL dz, unsigned int nx, unsigned int ny, unsigned int nz)
{
  unsigned int i, j, k, o, xy;
  xy = nx*ny;

  REAL err = 0., l1_norm = 0., l2_norm = 0., linf_norm = 0.;

  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {

        err = (exp(-3*M_PI*M_PI*t)*SINE_DISTRIBUTION(i,j,k,dx,dy,dz)) - u[i+nx*j+xy*k];
        
        l1_norm += fabs(err);
        l2_norm += err*err;
        linf_norm = fmax(linf_norm,fabs(err));
      }
    }
  }
  
  printf("L1 norm                                       :  %e\n", dx*dy*dz*l1_norm);
  printf("L2 norm                                       :  %e\n", l2_norm);
  printf("Linf norm                                     :  %e\n", linf_norm);
}

/*****************/
/* PRINT SUMMARY */
/*****************/
void PrintSummary(const char* kernelName, const char* optimization,
  REAL computeTimeInSeconds, float gflops, REAL outputTimeInSeconds, 
  const int computeIterations, const int nx, const int ny, const int nz)
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