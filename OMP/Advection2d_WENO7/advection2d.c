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
REAL FDM_7_Reconstruct1d(
                  const REAL vmmm,
                  const REAL vmm,
                  const REAL vm,
                  const REAL v,
                  const REAL vp,
                  const REAL vpp,
                  const REAL vppp,
                  const REAL ummm,
                  const REAL umm,
                  const REAL um,
                  const REAL u,
                  const REAL up,
                  const REAL upp,
                  const REAL uppp){
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
  hn = (-3*vmmm + 25*vmm - 101*vm  + 319*v + 214*vp - 38*vpp + 4*vppp)/420;

  // Numerical Flux at cell boundary, $v_{i+1/2}^{+}$;
  hp = (4*ummm - 38*umm  + 214*um  + 319*u - 101*up + 25*upp - 3*uppp)/420;
  
  // Compute the numerical flux v_{i+1/2}
  dflux = (hn+hp);
  return dflux;
}

/***********************/
/* WENO RECONSTRUCTION */
/***********************/
REAL WENO7_Reconstruct1d(
                  const REAL vmmm,
                  const REAL vmm,
                  const REAL vm,
                  const REAL v,
                  const REAL vp,
                  const REAL vpp,
                  const REAL vppp,
                  const REAL ummm,
                  const REAL umm,
                  const REAL um,
                  const REAL u,
                  const REAL up,
                  const REAL upp,
                  const REAL uppp){
  // *************************************************************************
  // Input: v(i) = [v(i-3) v(i-2) v(i-1) v(i) v(i+1) v(i+2) v(i+3)];
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
  //                               |_______________S3______________|
  //                               |                               |
  //                       |______________S2_______________|       |
  //                       |                               |       |
  //               |______________S1_______________|       |       |
  //               |                               |       |       |
  //       |_______________S0______________|       |       |       |
  //     ..|---o---|---o---|---o---|---o---|---o---|---o---|---o---|...
  //       | I{i-3}| I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}| I{i+3}|
  //                                      -|
  //                                     i+1/2
  //
  //       |______________S0_______________|
  //       |                               |
  //       |       |______________S1_______________|
  //       |       |                               |
  //       |       |       |______________S2_______________|
  //       |       |       |                               |
  //       |       |       |       |_______________S3______________|
  //     ..|---o---|---o---|---o---|---o---|---o---|---o---|---o---|...
  //       | I{i-3}| I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|| I{i+3}
  //                               |+
  //                             i-1/2
  //
  // WENO stencil: S{i} = [ I{i-3},...,I{i+3} ]
  // *************************************************************************
  REAL B0n, B1n, B2n, B3n, B0p, B1p, B2p, B3p;
  REAL w0n, w1n, w2n, w3n, w0p, w1p, w2p, w3p;
  REAL a0n, a1n, a2n, a3n, a0p, a1p, a2p, a3p;
  REAL alphasumn, alphasump, hn, hp, dflux;
  
  // Smooth Indicators (beta factors)
  B0n = vm*(134241*vm-114894*v) + vmmm*(56694*vm-47214*vmm+6649*vmmm-22778*v)
        +25729*v*v + vmm*(-210282*vm+85641*vmm+86214*v);
  B1n = v*(41001*v-30414*vp) + vmm*(-19374*vm+3169*vmm+19014*v-5978*vp)
        +6649*vp*vp + vm*(33441*vm-70602*v+23094*vp);
  B2n = vp*(33441*vp-19374*vpp) + vm*(6649*vm-30414*v+23094*vp-5978*vpp)
        +3169*vpp*vpp + v*(41001*v-70602*vp+19014*vpp);
  B3n = vpp*(85641*vpp-47214*vppp) + v*(25729*v-114894*vp+86214*vpp-22778*vppp)
        +6649*vppp*vppp + vp*(134241*vp-210282*vpp+56694*vppp);
  
  // Alpha weights
  a0n = D0N/((EPS + B0n)*(EPS + B0n));
  a1n = D1N/((EPS + B1n)*(EPS + B1n));
  a2n = D2N/((EPS + B2n)*(EPS + B2n));
  a3n = D3N/((EPS + B3n)*(EPS + B3n));
  alphasumn = a0n + a1n + a2n + a3n;
  
  // ENO stencils weigths
  w0n = a0n/alphasumn;
  w1n = a1n/alphasumn;
  w2n = a2n/alphasumn;
  w3n = a3n/alphasumn;
  
  // Numerical Flux at cell boundary, $v_{i+1/2}^{-}$;
  hn = (w0n*(-3*vmmm + 13*vmm - 23*vm  + 25*v   ) +
        w1n*( 1*vmm  -  5*vm  + 13*v   +  3*vp  ) +
        w2n*(-1*vm   +  7*v   +  7*vp  -  1*vpp ) + 
        w3n*( 3*v    + 13*vp  -  5*vpp +  1*vppp))/12;

  // Smooth Indicators (beta factors)
  B0p = um*(134241*um-114894*u) + ummm*(56694*um-47214*umm+6649*ummm-22778*u)
        +25729*u*u + umm*(-210282*um+85641*umm+86214*u);
  B1p = u*(41001*u-30414*up) + umm*(-19374*um+3169*umm+19014*u-5978*up)
        +6649*up*up + um*(33441*um-70602*u+23094*up);
  B2p = up*(33441*up-19374*upp) + um*(6649*um-30414*u+23094*up-5978*upp)
        +3169*upp*upp + u*(41001*u-70602*up+19014*upp);
  B3p = upp*(85641*upp-47214*uppp) + u*(25729*u-114894*up+86214*upp-22778*uppp)
        +6649*uppp*uppp + up*(134241*up-210282*upp+56694*uppp);

  // Alpha weights
  a0p = D0P/((EPS + B0p)*(EPS + B0p));
  a1p = D1P/((EPS + B1p)*(EPS + B1p));
  a2p = D2P/((EPS + B2p)*(EPS + B2p));
  a3p = D3P/((EPS + B3p)*(EPS + B3p));
  alphasump = a0p + a1p + a2p + a3p;
  
  // ENO stencils weigths
  w0p = a0p/alphasump;
  w1p = a1p/alphasump;
  w2p = a2p/alphasump;
  w3p = a3p/alphasump;

  // Numerical Flux at cell boundary, $v_{i+1/2}^{+}$;
  hp = (w0p*( 1*ummm - 5*umm + 13*um  + 3*u   ) +
        w1p*(-1*umm  + 7*um  +  7*u   - 1*up  ) +
        w2p*( 3*um   +13*u   -  5*up  + 1*upp ) + 
        w3p*(25*u   - 23*up  + 13*upp - 3*uppp))/12;
  
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
  REAL f1mmm, f1mm, f1m, f1, f1p, f1pp, f1ppp;
  REAL g1mmm, g1mm, g1m, g1, g1p, g1pp, g1ppp;

  // Indexes
  unsigned int i, j, o;
  
  #pragma omp parallel shared (u,Lu) private (j,f1mmm,f1mm,f1m,f1,f1p,f1pp,f1ppp,g1mmm,g1mm,g1m,g1,g1p,g1pp,g1ppp,fu,fu_old,fp,fp_old)  
  {
    #pragma omp for
    // Nonlinear Advection
    for (j = 3; j < ny-4; j++) {
        
      o=nx*j;
      // Old resulst arrays
      fu_old=0;
      fp_old=0;
      
      f1mmm= 0.5*(alpha*u[0+o]+alpha*u[0+o]); // node(i-3)
      f1mm = 0.5*(alpha*u[1+o]+alpha*u[1+o]); // node(i-2)
      f1m  = 0.5*(alpha*u[2+o]+alpha*u[2+o]); // node(i-1)
      f1   = 0.5*(alpha*u[3+o]+alpha*u[3+o]); // node( i )  immm--imm--im--i--ip--ipp--ippp
      f1p  = 0.5*(alpha*u[4+o]+alpha*u[4+o]); // node(i+1)
      f1pp = 0.5*(alpha*u[5+o]+alpha*u[5+o]); // node(i+2)
      
      g1mmm= 0.5*(alpha*u[1+o]-alpha*u[1+o]); // node(i-2)     
      g1mm = 0.5*(alpha*u[2+o]-alpha*u[2+o]); // node(i-1)
      g1m  = 0.5*(alpha*u[3+o]-alpha*u[3+o]); // node( i )      imm--im--i--ip--ipp--ippp-ipppp
      g1   = 0.5*(alpha*u[4+o]-alpha*u[4+o]); // node(i+1)
      g1p  = 0.5*(alpha*u[5+o]-alpha*u[5+o]); // node(i+2)
      g1pp = 0.5*(alpha*u[6+o]-alpha*u[6+o]); // node(i+3)
            
      for (i = 3; i < nx-4; i++) {
          
        // Compute and split fluxes
        f1ppp= 0.5*(alpha*u[i+3+o]+alpha*u[i+3+o]); // node(i+3)  ippp
        
        g1ppp= 0.5*(alpha*u[i+4+o]-alpha*u[i+4+o]); // node(i+4)  ipppp
        
        // Reconstruct
        fu = WENO7_Reconstruct1d(f1mmm,f1mm,f1m,f1,f1p,f1pp,f1ppp,g1mmm,g1mm,g1m,g1,g1p,g1pp,g1ppp);        
        //fu = FDM_7_Reconstruct1d(f1mmm,f1mm,f1m,f1,f1p,f1pp,f1ppp,g1mmm,g1mm,g1m,g1,g1p,g1pp,g1ppp);
        
        // Compute Lu = -dF/dx
        Lu[i+o]=-(fu-fu_old)/dx; // -dudx
        
        // Save old results
        fu_old=fu;
        
        f1mmm= f1mm;  // node(i-3)
        f1mm = f1m;   // node(i-2)
        f1m  = f1;    // node(i-1)
        f1   = f1p;   // node( i )    immm--imm--im--i--ip--ipp--ippp
        f1p  = f1pp;  // node(i+1)
        f1pp = f1ppp; // node(i+2)
        
        g1mmm= g1mm;  // node(i-2)        
        g1mm = g1m;   // node(i-1)
        g1m  = g1;    // node( i )    imm--im--i--ip--ipp--ippp-ipppp
        g1   = g1p;   // node(i+1)
        g1p  = g1pp;  // node(i+2)
        g1pp = g1ppp; // node(i+3)
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
  REAL f1mmm, f1mm, f1m, f1, f1p, f1pp, f1ppp;
  REAL g1mmm, g1mm, g1m, g1, g1p, g1pp, g1ppp;
  
  // Indexes
  unsigned int i, j;
  
  #pragma omp parallel shared (u,Lu) private (i,f1mmm,f1mm,f1m,f1,f1p,f1pp,f1ppp,g1mmm,g1mm,g1m,g1,g1p,g1pp,g1ppp,fu,fu_old,fp,fp_old)  
  {
    #pragma omp for
    // Nonlinear Advection
    for (i = 3; i < nx-4; i++) {
   
      // Old resulst arrays
      fu_old=0;
      fp_old=0;
      
      f1mmm= 0.5*(alpha*u[i+nx*0]+alpha*u[i+nx*0]); // node(i-3)
      f1mm = 0.5*(alpha*u[i+nx*1]+alpha*u[i+nx*1]); // node(i-2)
      f1m  = 0.5*(alpha*u[i+nx*2]+alpha*u[i+nx*2]); // node(i-1)
      f1   = 0.5*(alpha*u[i+nx*3]+alpha*u[i+nx*3]); // node( i )  immm--imm--im--i--ip--ipp--ippp
      f1p  = 0.5*(alpha*u[i+nx*4]+alpha*u[i+nx*4]); // node(i+1)
      f1pp = 0.5*(alpha*u[i+nx*5]+alpha*u[i+nx*5]); // node(i+2)
      
      g1mmm= 0.5*(alpha*u[i+nx*1]-alpha*u[i+nx*1]); // node(i-2)    
      g1mm = 0.5*(alpha*u[i+nx*2]-alpha*u[i+nx*2]); // node(i-1)
      g1m  = 0.5*(alpha*u[i+nx*3]-alpha*u[i+nx*3]); // node( i )      imm--im--i--ip--ipp--ippp-ipppp
      g1   = 0.5*(alpha*u[i+nx*4]-alpha*u[i+nx*4]); // node(i+1)
      g1p  = 0.5*(alpha*u[i+nx*5]-alpha*u[i+nx*5]); // node(i+2)
      g1pp = 0.5*(alpha*u[i+nx*6]-alpha*u[i+nx*6]); // node(i+3)
          
      for (j = 3; j < ny-4; j++) {
          
        // Compute and split fluxes
        f1ppp= 0.5*(alpha*u[i+nx*(j+3)]+alpha*u[i+nx*(j+3)]); // node(i+3)
        
        g1ppp= 0.5*(alpha*u[i+nx*(j+4)]-alpha*u[i+nx*(j+4)]); // node(i+4)
        
        // Reconstruct
        fu = WENO7_Reconstruct1d(f1mmm,f1mm,f1m,f1,f1p,f1pp,f1ppp,g1mmm,g1mm,g1m,g1,g1p,g1pp,g1ppp);
        //fu = FDM_7_Reconstruct1d(f1mmm,f1mm,f1m,f1,f1p,f1pp,f1ppp,g1mmm,g1mm,g1m,g1,g1p,g1pp,g1ppp);
        
        // Compute Lv = -dG/dy
        Lu[i+nx*j]-=(fu-fu_old)/dy; // -dudy
        
        // Save old results
        fu_old=fu;
        
        f1mmm= f1mm;  // node(i-3)
        f1mm = f1m;   // node(i-2)
        f1m  = f1;    // node(i-1)
        f1   = f1p;   // node( i )    immm--imm--im--i--ip--ipp--ippp
        f1p  = f1pp;  // node(i+1)
        f1pp = f1ppp; // node(i+2)
        
        g1mmm= g1mm;  // node(i-2)     
        g1mm = g1m;   // node(i-1)
        g1m  = g1;    // node( i )    imm--im--i--ip--ipp--ippp-ipppp
        g1   = g1p;   // node(i+1)
        g1p  = g1pp;  // node(i+2)
        g1pp = g1ppp; // node(i+3)
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
