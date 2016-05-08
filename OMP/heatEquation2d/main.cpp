
#include "heat2d.h"
#include <time.h>

int main(int argc, char** argv)
{
  REAL C;
  unsigned int L, W, Nx, Ny, max_iters, blockX, blockY;

  if (argc == 9)
  {
    C = atof(argv[1]); // conductivity, here it is assumed: Cx = Cy = C.
    L = atoi(argv[2]); // domain lenght
    W = atoi(argv[3]); // domain width 
    Nx = atoi(argv[4]); // number cells in x-direction
    Ny = atoi(argv[5]); // number cells in x-direction
    max_iters = atoi(argv[6]); // number of iterations / time steps
    blockX = atoi(argv[7]); // block size in the i-direction
    blockY = atoi(argv[8]); // block size in the j-direction
  }
  else
  {
    printf("Usage: %s diffCoef L W nx ny i block_x block_y\n", argv[0]);
    exit(1);
  }

  unsigned int tot_iters, R;
  REAL dx, dy, dt, kx, ky, tFinal;

  dx = (REAL)L/Nx; // dx, cell size
  dy = (REAL)W/Ny; // dy, cell size
  dt = 1/(2*C*(1/dx/dx+1/dy/dy)); // dt, fix time step size
  kx = C*dt/(dx*dx); // numerical conductivity
  ky = C*dt/(dy*dy); // numerical conductivity
  tFinal = dt*max_iters; printf("Final time: %g\n",tFinal);
  R = 1; // halo regions width (in cells size)

  // Initialize solution arrays
  REAL *h_u, *h_un; 
  REAL *d_u, *d_un;

  h_u  = (REAL*)malloc(sizeof(REAL)*(Nx+R)*(Ny+R));
  h_un = (REAL*)malloc(sizeof(REAL)*(Nx+R)*(Ny+R));

  // Set Domain Initial Condition and BCs
  Call_Init(4,h_u,dx,dy,Nx+R,Ny+R);

  // Request computer current time
  time_t t = clock();

  int UseSolverNo = 3;
  switch (UseSolverNo) {
    case 1: {
      printf("Using CPU solver with Dirichlet BCs \n");
      Call_CPU_Jacobi2d_v2(h_u,h_un,max_iters,kx,ky,Nx+R,Ny+R);
      break;
    }
    case 2: {
      printf("Using CPU solver with Neumann BCs \n");
      Call_CPU_Jacobi2d(h_u,h_un,max_iters,kx,ky,Nx+R,Ny+R);
      break;
    }
    case 3: {
      printf("Using OMP solver with Dirichlet BCs \n");
      Call_OMP_Jacobi2d_v2(h_u,h_un,max_iters,kx,ky,Nx+R,Ny+R);
      break;
    }
    case 4: {
      printf("Using OMP solver with Neumann BCs \n");
      Call_OMP_Jacobi2d(h_u,h_un,max_iters,kx,ky,Nx+R,Ny+R);
      break;
    }
  }
  // Measure and Report computation time
  t += clock(); REAL tCPU = (REAL)t/CLOCKS_PER_SEC;

  // uncomment to print solution to terminal
  if (DEBUG) print2D(h_un,Nx+R,Ny+R);

  float gflops = CalcGflops(tCPU, max_iters, Nx+R, Ny+R);
  PrintSummary("HeatEq2D (7-pt)", "none", tCPU, gflops, tFinal, max_iters, Nx+R, Ny+R);
  CalcError(h_un, tFinal, dx, dy, Nx+R, Ny+R);

  // Write solution to file
  if (WRITE) Save2D(h_un,Nx+R,Ny+R); 

  // Free memory on host and device
  free(h_u);
  free(h_un);

  return 0;
}
