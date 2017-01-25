//
//  main.c
//  Acoustics2d-CPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "advection2d.h"
#include <time.h>

int main(int argc, char** argv){
    
    // Set number of threads 
    omp_set_num_threads(8);

    // Input variables 
    REAL alpha, dt, dx, dy;
    unsigned int Nx, Ny, max_iter, blockX, blockY;

    if (argc == 10)
    {
        alpha = atof(argv[1]); // Pressure
        dt = atof(argv[2]); // The stability parameter
        dx = atof(argv[3]); // domain lenght
        dy = atof(argv[4]); // domain width
        Nx = atoi(argv[5]); // number cells in x-direction
        Ny = atoi(argv[6]); // number cells in y-direction
        max_iter=atoi(argv[7]);  // number of iterations / time steps
        blockX = atoi(argv[8]); // block size in the i-direction
        blockY = atoi(argv[9]); // block size in the j-direction
    }
    else
    {
        printf("Usage: %s alpha dt dx dy Nx Ny Iterations block_x block_y\n", argv[0]);
        exit(1);
    }

    // Define Constants
    const REAL   L   = dx*(Nx-1);     // dx, cell size
    const REAL   W   = dy*(Ny-1);     // dy, cell size
    const REAL tFinal= dt*max_iter;   // printf("Final time: %g\n",tFinal);
    
    // Feedback
    printf("alpha: %g\n",alpha);
    printf("dx: %g\n",dx);
    printf("dy: %g\n",dy);
    printf("dt: %g\n",dt);
    printf("tEnd: %g\n",tFinal);
    printf("Cuda Blocks Details:\n");
    printf("Threads in X: %d\n",blockX);
    printf("Threads in Y: %d\n",blockY);
    printf("\n");

    // Initialize solution arrays
    REAL *h_u;  h_u  = (REAL*)malloc(sizeof(REAL)*Nx*Ny);
    REAL *h_uo; h_uo = (REAL*)malloc(sizeof(REAL)*Nx*Ny);
    REAL *h_Lu; h_Lu = (REAL*)malloc(sizeof(REAL)*Nx*Ny);

    // Set Domain Initial Condition and BCs
    Call_Init2d(1,h_u,dx,dy,Nx,Ny);
    
    // Request computer current time
    time_t t = clock();
    
    // Call WENO-RK solver
    for(unsigned int iterations = 0; iterations < max_iter; iterations++)
    {
        // Runge Kutta Step 0
        memcpy(h_uo,h_u,sizeof(REAL)*Nx*Ny);
        
        // Set memory of temporal variables to zero
        memset(h_Lu,0,sizeof(REAL)*Nx*Ny);

        // Runge Kutta Step 1
        Compute_Adv_x(h_u,h_Lu,alpha,Nx,Ny,dx); //Print2D(h_dudx,Nx,Ny);
        Compute_Adv_y(h_u,h_Lu,alpha,Nx,Ny,dy); //Print2D(h_dudy,Nx,Ny);
        Compute_sspRK(h_u,h_uo,h_Lu,1,Nx,Ny,dt);
        
        // Runge Kutta Step 2
        Compute_Adv_x(h_u,h_Lu,alpha,Nx,Ny,dx); //Print2D(h_dudx,Nx,Ny);
        Compute_Adv_y(h_u,h_Lu,alpha,Nx,Ny,dy); //Print2D(h_dudy,Nx,Ny);
        Compute_sspRK(h_u,h_uo,h_Lu,2,Nx,Ny,dt);
        
        // Runge Kutta Step 3
        Compute_Adv_x(h_u,h_Lu,alpha,Nx,Ny,dx); //Print2D(h_dudx,Nx,Ny);
        Compute_Adv_y(h_u,h_Lu,alpha,Nx,Ny,dy); //Print2D(h_dudy,Nx,Ny);
        Compute_sspRK(h_u,h_uo,h_Lu,3,Nx,Ny,dt);
        
    }
    
    // Measure and Report computation time
    t += clock(); REAL tCPU = (REAL)t/CLOCKS_PER_SEC; printf("Computation took %lf seconds\n", tCPU);
    
    float gflops = CalcGflops(tCPU, max_iter, Nx, Ny);
    PrintSummary(argv[0], "none", tCPU, gflops, tFinal, max_iter, Nx, Ny);
    //CalcError(h_u, tFinal, dx, dy, Nx, Ny);

    // Write solution to file
    if (WRITE) SaveBinary2D(h_u,Nx,Ny);
    
    // Free memory on host and device
    free(h_u);
    free(h_uo);
    free(h_Lu);
    
    return 0;
}
