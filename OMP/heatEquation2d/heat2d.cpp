
#include "heat2d.h"
/*******************************/
/* Prints a flattened 2D array */
/*******************************/
void print2D(REAL *u, const unsigned int nx, const unsigned int ny)
{
  unsigned int i, j;

  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      printf("%8.2f", u[i+nx*j]);
    }
    printf("\n");
  }
}

/**************************/
/* Write to file 2D array */
/**************************/
void Save2D(REAL *u, const unsigned int nx, const unsigned int ny)
{
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");  
  if (pFile != NULL) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {      
        fprintf(pFile, "%d\t %d\t %g\n",j,i,u[i+nx*j]);
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
  const REAL dx, const REAL dy, 
  unsigned int nx, unsigned int ny)
{
  unsigned int i, j, o; 

  switch (IC) {
    case 1: {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        // set all domain's cells equal to zero
        o = i+nx*j;  u0[o] = 0.0;
        // set BCs in the domain 
        if (j==0)    u0[o] = 0.0; // bottom
        if (i==0)    u0[o] = 0.0; // left
        if (j==ny-1) u0[o] = 1.0; // top
        if (i==nx-1) u0[o] = 1.0; // right
      }
    }
    break;
  }
  case 2: {
    float u_bl = 0.7f;
    float u_br = 1.0f;
    float u_tl = 0.7f;
    float u_tr = 1.0f;

    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        // set all domain's cells equal to zero
        o = i+nx*j;  u0[o] = 0.0;
        // set BCs in the domain 
        if (j==0)    u0[o] = u_bl + (u_br-u_bl)*i/(nx+1); // bottom
        if (j==ny-1) u0[o] = u_tl + (u_tr-u_tl)*i/(nx+1); // top
        if (i==0)    u0[o] = u_bl + (u_tl-u_bl)*j/(ny+1); // left
        if (i==nx-1) u0[o] = u_br + (u_tr-u_br)*j/(ny+1); // right
      }
    }
    break;
  }
  case 3: {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        // set all domain's cells equal to zero
        o = i+nx*j;  u0[o] = 0.0;
        // set left wall to 1
        if (i==nx-1) u0[o] = 1.0;
      }
    }
    break;
  }
  case 4: {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        // set all domain's cells equal to :
        o = i+nx*j;  u0[o] = SINE_DISTRIBUTION(i,j,dx,dy); 
        // set BCs in the domain 
        if (i==0 || i==nx-1 || j==0 || j==ny-1) u0[o] = 0.0;
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
void Call_CPU_Jacobi2d(REAL *u,REAL *un, const unsigned int max_iters,
  const REAL kx, const REAL ky, const unsigned int nx, const unsigned int ny)
{
  // Using (i,j) = [i+N*j] indexes
  unsigned int i, j, k, o, n, s, e, w;

  for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
  {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {

          o = i +(nx*j);          // node( j,i,k )     n
        	n = (i==nx-1) ? o:o+nx; // node(j+1,i,k)     |
        	s = (i==0)    ? o:o-nx; // node(j-1,i,k)  w--o--e
        	e = (j==ny-1) ? o:o+1;  // node(j,i+1,k)     |
        	w = (j==0)    ? o:o-1;  // node(j,i-1,k)     s

        	un[o] = u[o] + kx*(u[e]-2*u[o]+u[w]) + ky*(u[n]-2*u[o]+u[s]);
      }
    } 
  SWAP(REAL*, u, un);
  }
}

void Call_CPU_Jacobi2d_v2(REAL *u, REAL *un, const unsigned int max_iters,
  const REAL kx, const REAL ky, const unsigned int nx, const unsigned int ny)
{
  // Using (i,j,k) = [i+N*j+M*N*k] indexes
  unsigned int i, j, k, o, n, s, e, w;

  for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
  {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {

        o = i+nx*j; // node( j,i )     n
        n = o+nx;   // node(j+1,i)     |
        s = o-nx;   // node(j-1,i)  w--o--e
        e = o+1;    // node(j,i+1)     |
        w = o-1;    // node(j,i-1)     s

        if (i>0 && i<nx-1 && j>0 && j<ny-1)
          un[o] = u[o] + kx*(u[e]-2*u[o]+u[w]) + ky*(u[n]-2*u[o]+u[s]);
        else
          un[o] = u[o];
      }
    }
  SWAP(REAL*, u, un);
  }
}

/************************/
/* JACOBI SOLVERS - OMP */
/************************/
void Call_OMP_Jacobi2d(REAL *u,REAL *un, const unsigned int max_iters,
  const REAL kx, const REAL ky, const unsigned int nx, const unsigned int ny)
{
  // Using (i,j,k) = [i+N*j+M*N*k] indexes
  unsigned int i, j, k, o, n, s, e, w;

  #pragma omp parallel default(shared) 
  {
    for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
    {
      #pragma omp for schedule(static)
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {

          o = i +(nx*j);          // node( j,i,k )     n
          n = (i==nx-1) ? o:o+nx; // node(j+1,i,k)     |
          s = (i==0)    ? o:o-nx; // node(j-1,i,k)  w--o--e
          e = (j==ny-1) ? o:o+1;  // node(j,i+1,k)     |
          w = (j==0)    ? o:o-1;  // node(j,i-1,k)     s

          un[o] = u[o] + kx*(u[e]-2*u[o]+u[w]) + ky*(u[n]-2*u[o]+u[s]);
        }
      }
      #pragma omp single
      {
        SWAP(REAL*, u, un);
      }
    }
  }
}

void Call_OMP_Jacobi2d_v2(REAL *u, REAL *un, const unsigned int max_iters,
  const REAL kx, const REAL ky, const unsigned int nx, const unsigned int ny)
{
  // Using (i,j,k) = [i+N*j+M*N*k] indexes
	unsigned int i, j, k, o, n, s, e, w;

	#pragma omp parallel default(shared) 
  {
		for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
    {
			#pragma omp for schedule(static)
			for (j = 0; j < ny; j++) {
				for (i = 0; i < nx; i++) {

          o = i+nx*j; // node( j,i )     n
          n = o+nx;   // node(j+1,i)     |
          s = o-nx;   // node(j-1,i)  w--o--e
          e = o+1;    // node(j,i+1)     |
          w = o-1;    // node(j,i-1)     s

          if (i>0 && i<nx-1 && j>0 && j<ny-1)
            un[o] = u[o] + kx*(u[e]-2*u[o]+u[w]) + ky*(u[n]-2*u[o]+u[s]);
          else
            un[o] = u[o];
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
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny)
{
    return iterations*(double)((nx*ny) * 1e-9 * FLOPS)/computeTimeInSeconds;
}

/***********************/
/* COMPUTE ERROR NORMS */
/***********************/
void CalcError(REAL *u, const REAL t, const REAL dx, const REAL dy, unsigned int nx, unsigned int ny)
{
  unsigned int i, j, o, xy;
  REAL err = 0., l1_norm = 0., l2_norm = 0., linf_norm = 0.;
 
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {

      err = (exp(-2*M_PI*M_PI*t)*SINE_DISTRIBUTION(i,j,dx,dy)) - u[i+nx*j];
      
      l1_norm += fabs(err);
      l2_norm += err*err;
      linf_norm = fmax(linf_norm,fabs(err));
    }
  }
  
  printf("L1 norm                                       :  %e\n", dx*dy*l1_norm);
  printf("L2 norm                                       :  %e\n", l2_norm);
  printf("Linf norm                                     :  %e\n", linf_norm);
}

/*****************/
/* PRINT SUMMARY */
/*****************/
void PrintSummary(const char* kernelName, const char* optimization,
  REAL computeTimeInSeconds, float gflops, REAL outputTimeInSeconds, 
  const int computeIterations, const int nx, const int ny)
{
  printf("=========================== %s =======================\n", kernelName);
  printf("Optimization                                 :  %s\n", optimization);
  printf("Kernel time ex. data transfers               :  %lf seconds\n", computeTimeInSeconds);
  printf("===================================================================\n");
  printf("Total effective GFLOPs                       :  %lf\n", gflops);
  printf("===================================================================\n");
  printf("2D Grid Size                                 :  %d x %d \n", nx,ny);
  printf("Iterations                                   :  %d\n", computeIterations);
  printf("Final Time                                   :  %g\n", outputTimeInSeconds);
  printf("===================================================================\n");
}
