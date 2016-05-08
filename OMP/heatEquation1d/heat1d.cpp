
#include "heat1d.h"
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

/**************************/
/* Write to file 1D array */
/**************************/
void Save1D(REAL *u, const unsigned int nx)
{
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");  
  if (pFile != NULL) {
    for (int i = 0; i < nx; i++) {      
      fprintf(pFile, "%d\t %g\n",i,u[i]);
  	}
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/

void Call_Init(const int IC, REAL *u0, const REAL dx, unsigned int nx)
{
  unsigned int i;

  switch (IC) {
    case 1: {
      // Uniform Temperature in the domain, temperature will be imposed at boundaries
      for (i = 0; i < nx; i++) u0[i]=0.0;
      // Set Dirichlet boundary conditions in global domain as u0[0]=0.0;  u0[NX]=1.0; namely
      u0[0]=0.0; u0[nx]=1.0;
      break;
    }
    case 2: {
      // A square jump problem
      for (i= 0; i < nx; i++) {if (i>0.3*nx && i<0.7*nx) u0[i]=1.0; else u0[i]=0.0;}
      // Set Neumann boundary conditions in global domain u0'[0]=0.0;  u0'[NX]=0.0;
      break;
    }
    case 3: {
      for (i = 0; i < nx; i++) {
        // set all domain's cells equal to :
        u0[i] = SINE_DISTRIBUTION(i,dx); 
        // set BCs in the domain 
        if (i==0 || i==nx-1) u0[i] = 0.0;
      }
      break;
    }
    // here to add another IC
  }
}

/************************/
/* JACOBI SOLVERS - CPU */
/************************/
void Call_CPU_Jacobi1d(REAL *u,REAL *un, const unsigned int max_iters, const REAL kx, const unsigned int nx)
{
  // Using (i) = [i] index
  unsigned int i, r, o, l;

  for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
  {
    for (i = 0; i < nx; i++) {

      o = i;                  // node( i )
    	r = (i==nx-1) ? o:o+1;  // node(i+1)  l---o---r
    	l = (i==0)    ? o:o-1;  // node(i-1)

    	un[o] = u[o] + kx*(u[r]-2*u[o]+u[l]);
    }
  SWAP(REAL*, u, un);
  }
}

void Call_CPU_Jacobi1d_v2(REAL *u, REAL *un, const unsigned int max_iters, const REAL kx, const unsigned int nx)
{
  // Using (i) = [i] index
  unsigned int i, r, o, l;

  for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
  {
    for (i = 0; i < nx; i++) {

      o = i;    // node( i )
      r = o+1;  // node(i+1)  l---o---r
      l = o-1;  // node(i-1)

      if (i>0 && i<nx-1)
        un[o] = u[o] + kx*(u[r]-2*u[o]+u[l]);
      else
        un[o] = u[o];
    }
  SWAP(REAL*, u, un);
  }
}

/************************/
/* JACOBI SOLVERS - OMP */
/************************/
void Call_OMP_Jacobi1d(REAL *u,REAL *un, const unsigned int max_iters, const REAL kx, const unsigned int nx)
{
  // Using (i) = [i] index
  unsigned int i, r, o, l;

  #pragma omp parallel default(shared) 
  {
    for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
    {
      #pragma omp for schedule(static)
        for (i = 0; i < nx; i++) {

        o = i;                  // node( i )
        r = (i==nx-1) ? o:o+1;  // node(i+1)  l---o---r
        l = (i==0)    ? o:o-1;  // node(i-1)

        un[o] = u[o] + kx*(u[r]-2*u[o]+u[l]);
        }
      #pragma omp single
      {
        SWAP(REAL*, u, un);
      }
    }
  }
}

void Call_OMP_Jacobi1d_v2(REAL *u, REAL *un, const unsigned int max_iters, const REAL kx, const unsigned int nx)
{
  // Using (i) = [i] index
  unsigned int i, r, o, l;

	#pragma omp parallel default(shared) 
  {
		for(unsigned int iterations = 1; iterations < max_iters; iterations++) 
    {
			#pragma omp for schedule(static)
				for (i = 0; i < nx; i++) {

          o = i;    // node( i )
          r = o+1;  // node(i+1)  l---o---r
          l = o-1;  // node(i-1)

          if (i>0 && i<nx-1)
            un[o] = u[o] + kx*(u[r]-2*u[o]+u[l]);
          else
            un[o] = u[o];
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
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx)
{
    return iterations*(double)((nx) * 1e-9 * FLOPS)/computeTimeInSeconds;
}

/***********************/
/* COMPUTE ERROR NORMS */
/***********************/
void CalcError(REAL *u, const REAL t, const REAL dx, unsigned int nx)
{
  unsigned int i;

  REAL err = 0., l1_norm = 0., l2_norm = 0., linf_norm = 0.;

  for (i = 0; i < nx; i++) {

    err = (exp(-M_PI*M_PI*t)*SINE_DISTRIBUTION(i,dx)) - u[i];
    
    l1_norm += fabs(err);
    l2_norm += err*err;
    linf_norm = fmax(linf_norm,fabs(err));
  }
  
  printf("L1 norm                                       :  %e\n", dx*l1_norm);
  printf("L2 norm                                       :  %e\n", l2_norm);
  printf("Linf norm                                     :  %e\n", linf_norm);
}

/*****************/
/* PRINT SUMMARY */
/*****************/
void PrintSummary(const char* kernelName, const char* optimization,
  REAL computeTimeInSeconds, float gflops, REAL outputTimeInSeconds, 
  const int computeIterations, const int nx)
{
  printf("=========================== %s =======================\n", kernelName);
  printf("Optimization                                 :  %s\n", optimization);
  printf("Kernel time ex. data transfers               :  %lf seconds\n", computeTimeInSeconds);
  printf("===================================================================\n");
  printf("Total effective GFLOPs                       :  %lf\n", gflops);
  printf("===================================================================\n");
  printf("1D Grid Size                                 :  %d\n", nx);
  printf("Iterations                                   :  %d\n", computeIterations);
  printf("Final Time                                   :  %g\n", outputTimeInSeconds);
  printf("===================================================================\n");
}