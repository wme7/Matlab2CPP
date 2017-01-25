//
//  advection2d.h
//  AcousticsNd-CPU
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#ifndef advection2d_h
#define advection2d_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* WENO constants */
#define D0N ((double)1)/35
#define D1N ((double)12)/35
#define D2N ((double)18)/35
#define D3N ((double)4)/35
#define D0P ((double)4)/35
#define D1P ((double)18)/35
#define D2P ((double)12)/35
#define D3P ((double)1)/35
#define EPS 1E-6

/* Math constants */
#define M_PI_2 1.57079632679489661923 // pi/2
#define M_PI 3.14159265358979323846 // pi
#define M_2_PI 0.636619772367581343076 // 2/pi

/* Write solution file */
#define DISPL 0 // Display all error messages
#define WRITE 1 // Write solution to file
#define FLOPS 8.0

/* Define macros */
#define I2D(n,i,j) ((i)+(n)*(j)) // transfrom a 2D array index pair into linear index memory
#define DIVIDE_INTO(x,y) (((x)+(y)-1)/(y)) // define No. of blocks/warps
#define SINE_DISTRIBUTION(i, dx) sin(M_2_PI*i*dx)
#define EXP_DISTRIBUTION(i, j, dx, dy, d, t0) exp( -((-5+i*dx)*(-5+i*dx)+(-5+j*dy)*(-5+j*dy))/(4*d*t0) )
#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

/* Use floats of doubles */
#define USE_FLOAT false // set false to use double
#if USE_FLOAT
	#define REAL	float
	#define MPI_CUSTOM_REAL MPI_REAL
#else
	#define REAL	double
	#define MPI_CUSTOM_REAL MPI_DOUBLE
#endif

/* Declare functions */
void Call_Init2d(const int IC, REAL *u, const REAL dx, const REAL dy, const unsigned int nx, const unsigned int ny);
void Compute_Adv_x(REAL *u, REAL *Lu, const REAL alpha, const unsigned int nx, const unsigned int ny, const REAL dx);
void Compute_Adv_y(REAL *u, REAL *Lu, const REAL alpha, const unsigned int nx, const unsigned int ny, const REAL dy);
void Compute_sspRK(REAL *u, const REAL *uo, const REAL *Lu, const int step, const unsigned int nx, const unsigned int ny, const REAL dt);

float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny);
void PrintSummary(const char* kernelName, const char* optimization, REAL computeTimeInSeconds, float gflops, REAL outputTimeInSeconds,const int computeIterations, const int nx, const int ny);
void CalcError(REAL *u, const REAL t, const REAL dx, const REAL dy, unsigned int nx, unsigned int ny);

void Print2D(REAL *u, const unsigned int nx, const unsigned int ny);
void Save2D(REAL *h_u, const unsigned int nx, const unsigned int ny);
void SaveBinary2D(REAL *h_u, const unsigned int nx, const unsigned int ny);

#endif /* advection2d_h */
