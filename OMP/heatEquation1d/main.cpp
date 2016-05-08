
#include "heat1d.h"
#include <time.h>

int main(int argc, char** argv)
{
  REAL C;
  unsigned int L, Nx, max_iters, blockX;

  if (argc == 6)
  {
    C = atof(argv[1]); // conductivity, here it is assumed: Cx = Cy = Cz = C.
    L = atoi(argv[2]); // domain lenght
    Nx = atoi(argv[3]); // number cells in x-direction
    max_iters = atoi(argv[4]); // number of iterations / time steps
    blockX = atoi(argv[5]); // block size in the i-direction
  }
  else
  {
    printf("Usage: %s diffCoef L nx i block_x\n", argv[0]);
    exit(1);
  }

  unsigned int tot_iters, R;
  REAL dx, dt, kx, tFinal;

  dx = (REAL)L/Nx; // dx, cell size
  dt = 1/(2*C*(1/dx/dx)); // dt, fix time step size
  kx = C*dt/(dx*dx); // numerical conductivity
  tFinal = dt*max_iters; printf("Final time: %g\n",tFinal);
  R = 1; // halo regions width (in cells size)

  // Initialize solution arrays
  REAL *h_u, *h_un; 
  REAL *d_u, *d_un;

  h_u  = (REAL*)malloc(sizeof(REAL)*(Nx+R));
  h_un = (REAL*)malloc(sizeof(REAL)*(Nx+R));

  // Set Domain Initial Condition and BCs
  Call_Init(3,h_u,dx,Nx+R);

  // Request computer current time
  time_t t = clock();

  int UseSolverNo = 3;
  switch (UseSolverNo) {
    case 1: {
      printf("Using CPU solver with Dirichlet BCs \n");
      Call_CPU_Jacobi1d_v2(h_u,h_un,max_iters,kx,Nx+R);
      break;
    }
    case 2: {
      printf("Using CPU solver with Neumann BCs \n");
      Call_CPU_Jacobi1d(h_u,h_un,max_iters,kx,Nx+R);
      break;
    }
    case 3: {
      printf("Using OMP solver with Dirichlet BCs \n");
      Call_OMP_Jacobi1d_v2(h_u,h_un,max_iters,kx,Nx+R);
      break;
    }
    case 4: {
      printf("Using OMP solver with Neumann BCs \n");
      Call_OMP_Jacobi1d(h_u,h_un,max_iters,kx,Nx+R);
      break;
    }
  }
  // Measure and Report computation time
  t += clock(); REAL tCPU = (REAL)t/CLOCKS_PER_SEC;

  // uncomment to print solution to terminal
  if (DEBUG) print1D(h_un,Nx+R);

  float gflops = CalcGflops(tCPU, max_iters, Nx+R);
  PrintSummary("HeatEq1D (7-pt)", "none", tCPU, gflops, tFinal, max_iters, Nx+R);
  CalcError(h_un, tFinal, dx, Nx+R);

  // Write solution to file
  if (WRITE) Save1D(h_un,Nx+R); 

  // Free memory on host and device
  free(h_u);
  free(h_un);

  return 0;
}
