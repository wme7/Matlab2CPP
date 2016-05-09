
#include "heat3d.h"
#include <time.h>

int main(int argc, char** argv)
{
  REAL C;
  unsigned int L, W, H, Nx, Ny, Nz, max_iters, blockX, blockY, blockZ;

  if (argc == 12)
  {
    C = atof(argv[1]); // conductivity, here it is assumed: Cx = Cy = Cz = C.
    L = atoi(argv[2]); // domain lenght
    W = atoi(argv[3]); // domain width 
    H = atoi(argv[4]); // domain height
    Nx = atoi(argv[5]); // number cells in x-direction
    Ny = atoi(argv[6]); // number cells in x-direction
    Nz = atoi(argv[7]); // number cells in x-direction
    max_iters = atoi(argv[8]); // number of iterations / time steps
    blockX = atoi(argv[9]); // block size in the i-direction
    blockY = atoi(argv[10]); // block size in the j-direction
    blockZ = atoi(argv[11]); // block size in the k-direction
  }
  else
  {
    printf("Usage: %s diffCoef L W H nx ny nz i block_x block_y block_z\n", argv[0]);
    exit(1);
  }

  unsigned int tot_iters, R;
  REAL dx, dy, dz, dt, kx, ky, kz, tFinal;

  dx = (REAL)L/(Nx-1); // dx, cell size
  dy = (REAL)W/(Ny-1); // dy, cell size
  dz = (REAL)H/(Nz-1); // dz, cell size
  dt = 1/(2*C*(1/dx/dx+1/dy/dy+1/dz/dz)); // dt, fix time step size
  kx = C*dt/(dx*dx); // numerical conductivity
  ky = C*dt/(dy*dy); // numerical conductivity
  kz = C*dt/(dz*dz); // numerical conductivity
  tFinal = dt*max_iters; printf("Final time: %g\n",tFinal);
  R = 1; // halo regions width (in cells size)

  // Initialize solution arrays
  REAL *h_u, *h_un; 
  REAL *d_u, *d_un;

  h_u  = (REAL*)malloc(sizeof(REAL)*Nx*Ny*Nz);
  h_un = (REAL*)malloc(sizeof(REAL)*Nx*Ny*Nz);

  // Set Domain Initial Condition and BCs
  Call_Init(3,h_u,dx,dy,dz,Nx,Ny,Nz);

  // Request computer current time
  time_t t = clock();

  int UseSolverNo = 3;
  switch (UseSolverNo) {
    case 1: {
      printf("Using CPU solver with Dirichlet BCs \n");
      Call_CPU_Jacobi3d_v2(h_u,h_un,max_iters,kx,ky,kz,Nx,Ny,Nz);
      break;
    }
    case 2: {
      printf("Using CPU solver with Neumann BCs \n");
      Call_CPU_Jacobi3d(h_u,h_un,max_iters,kx,ky,kz,Nx,Ny,Nz);
      break;
    }
    case 3: {
      printf("Using OMP solver with Dirichlet BCs \n");
      Call_OMP_Jacobi3d_v2(h_u,h_un,max_iters,kx,ky,kz,Nx,Ny,Nz);
      break;
    }
    case 4: {
      printf("Using OMP solver with Neumann BCs \n");
      Call_OMP_Jacobi3d(h_u,h_un,max_iters,kx,ky,kz,Nx,Ny,Nz);
      break;
    }
  }
  // Measure and Report computation time
  t += clock(); REAL tCPU = (REAL)t/CLOCKS_PER_SEC;

  // uncomment to print solution to terminal
  if (DEBUG) print3D(h_un,Nx,Ny,Nz);

  float gflops = CalcGflops(tCPU, max_iters, Nx, Ny, Nz);
  PrintSummary("HeatEq3D (7-pt)", "none", tCPU, gflops, tFinal, max_iters, Nx, Ny, Nz);
  CalcError(h_un, tFinal, dx, dy, dz, Nx, Ny, Nz);

  // Write solution to file
  if (WRITE) Save3D(h_un,Nx,Ny,Nz); 

  // Free memory on host and device
  free(h_u);
  free(h_un);

  return 0;
}
