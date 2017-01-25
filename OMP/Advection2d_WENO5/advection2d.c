//
//  acoustics.c
//  AcousticsNd-CPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "advection2d.h"

/*******************************/
/* Prints a flattened 1D array */
/*******************************/
void Print2D(REAL *u, const unsigned int nx, const unsigned int ny)
{
    unsigned int i, j;
    // print a single property on terminal
    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            printf("%8.2f", u[i+nx*j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**************************/
/* Write to file 1D array */
/**************************/
void Save2D(REAL *u, const unsigned int nx, const unsigned int ny)
{
    unsigned int i, j;
    // print result to txt file
    FILE *pFile = fopen("result.txt", "w");
    if (pFile != NULL) {
        for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++) {
                //fprintf(pFile, "%d\t %d\t %g\t %g\t %g\n",j,i,u[i+nx*j],v[i+nx*j],p[i+nx*j]);
		fprintf(pFile, "%g\n",u[i+nx*j]);
            }
        }
        fclose(pFile);
    } else {
        printf("Unable to save to file\n");
    }
}

void SaveBinary2D(REAL *u, const unsigned int nx, const unsigned int ny)
{
    /* NOTE: We save our result as float values always!
     *
     * In Matlab, the results can be loaded by simply doing 
     * fID = fopen('result.bin');
     * result = fread(fID,[4,nx*ny],'float')';
     * myplot(result,nx,ny);
     */

    float data;
    unsigned int i, j, k, xy, o;
    // print result to txt file
    FILE *pFile = fopen("result.bin", "w");
    if (pFile != NULL) {
        for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++) {
                o = i+nx*j; // index
                data = (float)u[o]; fwrite(&data,sizeof(float),1,pFile);
            }
        }
        fclose(pFile);
    } else {
        printf("Unable to save to file\n");
    }
}

/***************************/
/* PRESSURE INITIALIZATION */
/***************************/
void Call_Init2d(const int IC, REAL *u0, const REAL dx, const REAL dy, unsigned int nx, unsigned int ny)
{
    unsigned int i, j, o;
    
    switch (IC) {
        case 1: {
          // A square jump problem
          for (j= 0; j < ny; j++) {
            for (i= 0; i < nx; i++) {
              o = i+nx*j;
              if (i>0.4*nx && i<0.6*nx && j>0.4*ny && j<0.6*ny) {
                  u0[o]=10.0E1;
              } else {
                  u0[o]=0.0;
              }
            }
          }
          // Set Neumann boundary conditions in global domain u0'[0]=0.0;  u0'[NX]=0.0;
          break;
        }
        case 2: {
          // Homogeneous IC
          for (j= 0; j < ny; j++) {
            for (i= 0; i < nx; i++) {
              o = i+nx*j;
              u0[o]=0.0;
            }
          }
          break;
        }
        // here to add another IC
    }
}

/***********************/
/* FDM RECONSTRUCTIONS */
/***********************/
REAL FDM_5_Reconstruct1d(
                  const REAL vmm,
                  const REAL vm,
                  const REAL v,
                  const REAL vp,
                  const REAL vpp,
                  const REAL umm,
                  const REAL um,
                  const REAL u,
                  const REAL up,
                  const REAL upp){
  // *************************************************************************
  // Input: v(i) = [v(i-2) v(i-1) v(i) v(i+1) v(i+2) v(i+3)];
  // Output: res = df/dx;
  //
  // Based on:
  // C.W. Shu's Lectures notes on: 'ENO and WENO schemes for Hyperbolic
  // Conservation Laws'
  //
  // coded by Manuel Diaz, 02.10.2012, NTU Taiwan.
  // *************************************************************************
  //
  // Domain cells (I{i}) reference:
  //
  //                |           |   u(i)    |           |
  //                |  u(i-1)   |___________|           |
  //                |___________|           |   u(i+1)  |
  //                |           |           |___________|
  //             ...|-----0-----|-----0-----|-----0-----|...
  //                |    i-1    |     i     |    i+1    |
  //                |-         +|-         +|-         +|
  //              i-3/2       i-1/2       i+1/2       i+3/2
  //
  // ENO stencils (S{r}) reference:
  //
  //                           |___________S2__________|
  //                           |                       |
  //                   |___________S1__________|       |
  //                   |                       |       |    using only f^{+}
  //           |___________S0__________|       |       |
  //         ..|---o---|---o---|---o---|---o---|---o---|...
  //           | I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|
  //                                  -|
  //                                 i+1/2
  //
  //                   |___________S0__________|
  //                   |                       |
  //                   |       |___________S1__________|    using only f^{-}
  //                   |       |                       |
  //                   |       |       |___________S2__________|
  //                 ..|---o---|---o---|---o---|---o---|---o---|...
  //                   | I{i-1}|  I{i} | I{i+1}| I{i+2}| I{i+3}|
  //                                   |+
  //                                 i+1/2
  //
  // WENO stencil: S{i} = [ I{i-2},...,I{i+3} ]
  // *************************************************************************
  REAL hn, hp, dflux;
    
  // Numerical Flux at cell boundary, $v_{i+1/2}^{-}$;
  hn = ( 2*vmm - 13*vm + 47*v + 27*vp - 3*vpp)/60;

  // Numerical Flux at cell boundary, $v_{i+1/2}^{+}$;
  hp = (-3*umm + 27*um + 47*u - 13*up + 2*upp)/60;
  
  // Compute the numerical flux v_{i+1/2}
  dflux = (hn+hp);
  return dflux;
}

/***********************/
/* WENO RECONSTRUCTION */
/***********************/
REAL WENO5_Reconstruct1d(
                  const REAL vmm,
                  const REAL vm,
                  const REAL v,
                  const REAL vp,
                  const REAL vpp,
                  const REAL umm,
                  const REAL um,
                  const REAL u,
                  const REAL up,
                  const REAL upp){
  // *************************************************************************
  // Input: v(i) = [v(i-2) v(i-1) v(i) v(i+1) v(i+2) v(i+3)];
  // Output: res = df/dx;
  //
  // Based on:
  // C.W. Shu's Lectures notes on: 'ENO and WENO schemes for Hyperbolic
  // Conservation Laws'
  //
  // coded by Manuel Diaz, 02.10.2012, NTU Taiwan.
  // *************************************************************************
  //
  // Domain cells (I{i}) reference:
  //
  //                |           |   u(i)    |           |
  //                |  u(i-1)   |___________|           |
  //                |___________|           |   u(i+1)  |
  //                |           |           |___________|
  //             ...|-----0-----|-----0-----|-----0-----|...
  //                |    i-1    |     i     |    i+1    |
  //                |-         +|-         +|-         +|
  //              i-3/2       i-1/2       i+1/2       i+3/2
  //
  // ENO stencils (S{r}) reference:
  //
  //                           |___________S2__________|
  //                           |                       |
  //                   |___________S1__________|       |
  //                   |                       |       |    using only f^{+}
  //           |___________S0__________|       |       |
  //         ..|---o---|---o---|---o---|---o---|---o---|...
  //           | I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|
  //                                  -|
  //                                 i+1/2
  //
  //                   |___________S0__________|
  //                   |                       |
  //                   |       |___________S1__________|    using only f^{-}
  //                   |       |                       |
  //                   |       |       |___________S2__________|
  //                 ..|---o---|---o---|---o---|---o---|---o---|...
  //                   | I{i-1}|  I{i} | I{i+1}| I{i+2}| I{i+3}|
  //                                   |+
  //                                 i+1/2
  //
  // WENO stencil: S{i} = [ I{i-2},...,I{i+3} ]
  // *************************************************************************
  REAL B0n, B1n, B2n, B0p, B1p, B2p;
  REAL w0n, w1n, w2n, w0p, w1p, w2p;
  REAL a0n, a1n, a2n, a0p, a1p, a2p;
  REAL alphasumn, alphasump, hn, hp;
  REAL dflux;
  
  // Smooth Indicators (beta factors)
  B0n = C1312*(vmm-2*vm+v  )*(vmm-2*vm+v  ) + C14*(vmm-4*vm+3*v)*(vmm-4*vm+3*v);
  B1n = C1312*(vm -2*v +vp )*(vm -2*v +vp ) + C14*(vm-vp)*(vm-vp);
  B2n = C1312*(v  -2*vp+vpp)*(v  -2*vp+vpp) + C14*(3*v-4*vp+vpp)*(3*v-4*vp+vpp);
  
  // Alpha weights
  a0n = D0N/((EPS + B0n)*(EPS + B0n));
  a1n = D1N/((EPS + B1n)*(EPS + B1n));
  a2n = D2N/((EPS + B2n)*(EPS + B2n));
  alphasumn = a0n + a1n + a2n;
  
  // ENO stencils weigths
  w0n = a0n/alphasumn;
  w1n = a1n/alphasumn;
  w2n = a2n/alphasumn;
  
  // Numerical Flux at cell boundary, $v_{i+1/2}^{-}$;
  hn = (w0n*(2*vmm- 7*vm + 11*v) +
        w1n*( -vm + 5*v  + 2*vp) +
        w2n*( 2*v + 5*vp - vpp ))/6;

  // Smooth Indicators (beta factors)
  B0p = C1312*(umm-2*um+u  )*(umm-2*um +u  ) + C14*(umm-4*um+3*u)*(umm-4*um+3*u);
  B1p = C1312*(um -2*u +up )*(um -2*u  +up ) + C14*(um-up)*(um-up);
  B2p = C1312*(u  -2*up+upp)*(u  -2*up +upp) + C14*(3*u-4*up+upp)*(3*u-4*up+upp);
  
  // Alpha weights
  a0p = D0P/((EPS + B0p)*(EPS + B0p));
  a1p = D1P/((EPS + B1p)*(EPS + B1p));
  a2p = D2P/((EPS + B2p)*(EPS + B2p));
  alphasump = a0p + a1p + a2p;
  
  // ENO stencils weigths
  w0p = a0p/alphasump;
  w1p = a1p/alphasump;
  w2p = a2p/alphasump;

  // Numerical Flux at cell boundary, $v_{i+1/2}^{+}$;
  hp = (w0p*( -umm + 5*um + 2*u  ) +
        w1p*( 2*um + 5*u  - up   ) +
        w2p*(11*u  - 7*up + 2*upp))/6;
  
  // Compute the numerical flux v_{i+1/2}
  dflux = (hn+hp);
  return dflux;
}

/*****************/
/* Compute dF/dx */ // <==== parallel strategy: compute serialy by rows or by columns!
/*****************/
void Compute_Adv_x(
  REAL *u, 
  REAL *Lu, 
  const REAL alpha,
  const unsigned int nx, 
  const unsigned int ny, 
  const REAL dx)
{
  // Temporary variables
  REAL fu, fu_old, fp, fp_old;
  REAL f1mm, f1m, f1, f1p, f1pp;
  REAL g1mm, g1m, g1, g1p, g1pp;

  // Indexes
  unsigned int i, j, o;
  
  #pragma omp parallel shared (u,Lu) private (j,f1mm,f1m,f1,f1p,f1pp,g1mm,g1m,g1,g1p,g1pp,fu,fu_old,fp,fp_old)  
  {
    #pragma omp for
    // Nonlinear Advection
    for (j = 2; j < ny-3; j++) {
        
      o=nx*j;
      // Old resulst arrays
      fu_old=0;
      fp_old=0;
      
      f1mm= 0.5*(alpha*u[0+o]+alpha*u[0+o]); // node(i-2)
      f1m = 0.5*(alpha*u[1+o]+alpha*u[1+o]); // node(i-1)
      f1  = 0.5*(alpha*u[2+o]+alpha*u[2+o]); // node( i )  imm--im--i--ip--ipp
      f1p = 0.5*(alpha*u[3+o]+alpha*u[3+o]); // node(i+1)
           
      g1mm= 0.5*(alpha*u[1+o]-alpha*u[1+o]); // node(i-1)
      g1m = 0.5*(alpha*u[2+o]-alpha*u[2+o]); // node( i )      im--i--ip--ipp--ippp
      g1  = 0.5*(alpha*u[3+o]-alpha*u[3+o]); // node(i+1)
      g1p = 0.5*(alpha*u[4+o]-alpha*u[4+o]); // node(i+2)
            
      for (i = 2; i < nx-3; i++) {
          
        // Compute and split fluxes
        f1pp= 0.5*(alpha*u[i+2+o]+alpha*u[i+2+o]); // node(i+2)  ipp
        
        g1pp= 0.5*(alpha*u[i+3+o]-alpha*u[i+3+o]); // node(i+3)  ippp
        
        // Reconstruct
        fu = WENO5_Reconstruct1d(f1mm,f1m,f1,f1p,f1pp,g1mm,g1m,g1,g1p,g1pp);        
        //fu = FDM_5_Reconstruct1d(f1mm,f1m,f1,f1p,f1pp,g1mm,g1m,g1,g1p,g1pp);
        
        // Compute Lu = -dF/dx
        Lu[i+o]=-(fu-fu_old)/dx; // -dudx
        
        // Save old results
        fu_old=fu;
        
        f1mm= f1m;   // node(i-2)
        f1m = f1;    // node(i-1)
        f1  = f1p;   // node( i )    imm--im--i--ip--ipp
        f1p = f1pp;  // node(i+1)
                
        g1mm= g1m;   // node(i-1)
        g1m = g1;    // node( i )    im--i--ip--ipp--ippp
        g1  = g1p;   // node(i+1)
        g1p = g1pp;  // node(i+2)
      }
    }
  }
}

/*****************/
/* Compute dG/dx */ // <==== parallel strategy: compute serialy by rows or by columns!
/*****************/
void Compute_Adv_y(
  REAL *u, 
  REAL *Lu, 
  const REAL alpha, 
  const unsigned int nx, 
  const unsigned int ny, 
  const REAL dy)
{
  // Temporary variables
  REAL fu, fu_old, fp, fp_old;
  REAL f1mm, f1m, f1, f1p, f1pp;
  REAL g1mm, g1m, g1, g1p, g1pp;
  
  // Indexes
  unsigned int i, j;
  
  #pragma omp parallel shared (u,Lu) private (i,f1mm,f1m,f1,f1p,f1pp,g1mm,g1m,g1,g1p,g1pp,fu,fu_old,fp,fp_old)  
  {
    #pragma omp for
    // Nonlinear Advection
    for (i = 2; i < nx-3; i++) {
     
      // Old resulst arrays
      fu_old=0;
      fp_old=0;
      
      f1mm= 0.5*(alpha*u[i+nx*0]+alpha*u[i+nx*0]); // node(i-2)
      f1m = 0.5*(alpha*u[i+nx*1]+alpha*u[i+nx*1]); // node(i-1)
      f1  = 0.5*(alpha*u[i+nx*2]+alpha*u[i+nx*2]); // node( i )  imm--im--i--ip--ipp
      f1p = 0.5*(alpha*u[i+nx*3]+alpha*u[i+nx*3]); // node(i+1)
          
      g1mm= 0.5*(alpha*u[i+nx*1]-alpha*u[i+nx*1]); // node(i-1)
      g1m = 0.5*(alpha*u[i+nx*2]-alpha*u[i+nx*2]); // node( i )      im--i--ip--ipp--ippp
      g1  = 0.5*(alpha*u[i+nx*3]-alpha*u[i+nx*3]); // node(i+1)
      g1p = 0.5*(alpha*u[i+nx*4]-alpha*u[i+nx*4]); // node(i+2)
          
      for (j = 2; j < ny-3; j++) {
          
        // Compute and split fluxes
        f1pp= 0.5*(alpha*u[i+nx*(j+2)]+alpha*u[i+nx*(j+2)]); // node(i+2)
        
        g1pp= 0.5*(alpha*u[i+nx*(j+3)]-alpha*u[i+nx*(j+3)]); // node(i+3)
        
        // Reconstruct
        fu = WENO5_Reconstruct1d(f1mm,f1m,f1,f1p,f1pp,g1mm,g1m,g1,g1p,g1pp);
        //fu = FDM_5_Reconstruct1d(f1mm,f1m,f1,f1p,f1pp,g1mm,g1m,g1,g1p,g1pp);
        
        // Compute Lv = -dG/dy
        Lu[i+nx*j]-=(fu-fu_old)/dy; // -dudy
        
        // Save old results
        fu_old=fu;
        
        f1mm= f1m;   // node(i-2)
        f1m = f1;    // node(i-1)
        f1  = f1p;   // node( i )    imm--im--i--ip--ipp
        f1p = f1pp;  // node(i+1)
             
        g1mm= g1m;   // node(i-1)
        g1m = g1;    // node( i )    im--i--ip--ipp--ippp
        g1  = g1p;   // node(i+1)
        g1p = g1pp;  // node(i+2)
      }
    }
  }
}

/******************************/
/* Cartesian Laplace Operator */
/******************************/
void Compute_Diff_(
  REAL *u, 
  REAL *Lu, 
  const REAL diff, 
  const unsigned int nx, 
  const unsigned int ny)
{
  // Using (i,j,k) = [i+N*j+M*N*k] indexes
  unsigned int i, j, o, n, s, e, w, nn, ss, ee, ww;

  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {

      o = i+nx*j; // node( j,i )        nn
      nn= o+nx+nx;// node(j+2,i)         |
      n = o+nx;   // node(j+1,i)         n
      s = o-nx;   // node(j-1,i)         |
      ss= o-nx-nx;// node(j-2,i)  ww--w--o--e--ee
      ee= o+2;    // node(j,i+2)         |
      e = o+1;    // node(j,i+1)         s
      w = o-1;    // node(j,i-1)         |
      ww= o-2;    // node(j,i-2)         ss

      if (i>1 && i<nx-2 && j>1 && j<ny-2)
        Lu[o] += diff*(-u[ee]-u[nn]+16*u[n]+16*u[e]-60*u[o]+16*u[w]+16*u[s]-u[ss]-u[ww]);
    }
  }
}

/***********************/
/* Runge Kutta Methods */  // <==== this is perfectly parallel!
/***********************/
void Compute_sspRK(
  REAL *u, 
  const REAL *uo, 
  const REAL *Lu, 
  const int step, 
  const unsigned int nx, 
  const unsigned int ny, 
  const REAL dt)
{
  unsigned int i, j, o;
  // Compute Runge-Kutta step
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      // compute single index
      o=i+nx*j;
      // update only internal cells
      if (i>2 && i<nx-3 && j>2 && j<ny-3)
      {
        switch (step) {
          case 1: // step 1
            u[o] = uo[o]+dt*Lu[o];
            break;
          case 2: // step 2
            u[o] = 0.75*uo[o]+0.25*(u[o]+dt*Lu[o]);
            break;
          case 3: // step 3
            u[o] = (uo[o]+2*(u[o]+dt*Lu[o]))/3;
            break;
          }
      }
      //else do nothing!
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

      //err = (exp(-2*M_PI*M_PI*t)*SINE_DISTRIBUTION(i,j,dx,dy)) - u[i+nx*j];
      err = ((0.1/t)*EXP_DISTRIBUTION(i,j,dx,dy,1.0,t)) - u[i];
      
      l1_norm += fabs(err);
      l2_norm += err*err;
      linf_norm = fmax(linf_norm,fabs(err));
    }
  }
  
  printf("L1 norm                                       :  %e\n", dx*dy*l1_norm);
  printf("L2 norm                                       :  %e\n", sqrt(dx*dy*l2_norm));
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
}
