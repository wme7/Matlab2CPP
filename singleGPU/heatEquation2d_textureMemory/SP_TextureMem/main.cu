#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DEBUG 0 // Display all error messages
#define NX 1024 // number of cells in the x-direction
#define NY 1024 // number of cells in the y-direction
#define L 10.0 // domain length
#define W 10.0 // domain width
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 1.0 // tEnd, output time
#define DX (L/NX) // dx, cell size
#define DY (W/NY) // dy, cell size
#define DT (1/(2*C*(1/DX/DX+1/DY/DY))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define KY (C*DT/(DY*DY)) // numerical conductivity
#define NO_STEPS (TEND/DT) // No. of time steps
#define PI 3.1415926535897932f
#define COMPARE 0 // compare to CPU solution

#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_Y 2

#define DIVIDE_INTO(x,y) (((x)+(y)-1)/(y)) // define No. of blocks/warps

// Initialize Textures
texture<float, 2, cudaReadModeElementType>  tex_T;
texture<float, 2, cudaReadModeElementType>  tex_T_old;

/*********************************************/
/* JACOBI ITERATION FUNCTION - GPU - TEXTURE */
/*********************************************/
__global__ void Laplace2d_texture(float * __restrict__ un, const bool flag) {
    
  float o, n, s, e, w;
  //Threads id
  const int i = blockIdx.x * blockDim.x + threadIdx.x ;
  const int j = blockIdx.y * blockDim.y + threadIdx.y ;

  if (flag) {
    o = tex2D(tex_T_old,i, j ); // node( i,j )     n
    n = tex2D(tex_T_old,i,j+1); // node(i,j+1)     |
    s = tex2D(tex_T_old,i,j-1); // node(i,j-1)  w--o--e
    e = tex2D(tex_T_old,i+1,j); // node(i+1,j)     |
    w = tex2D(tex_T_old,i-1,j); // node(i-1,j)     s
  } else {
    o = tex2D(tex_T,i, j ); // node( i,j )     n
    n = tex2D(tex_T,i,j+1); // node(i,j+1)     |
    s = tex2D(tex_T,i,j-1); // node(i,j-1)  w--o--e
    e = tex2D(tex_T,i+1,j); // node(i+1,j)     |
    w = tex2D(tex_T,i-1,j); // node(i-1,j)     s
  }
  // --- Only update "interior" (not boundary) node points
  if (i>0 && i<NX-1 && j>0 && j<NY-1) un[i+j*NX] = o + KX*(e-2*o+w) + KY*(n-2*o+s);
}

/***********************************/
/* JACOBI ITERATION FUNCTION - CPU */
/***********************************/
void Laplace2d(float * __restrict u,float * __restrict un){
  // Using (i,j) = [i+N*j] indexes
  int i, j, o, n, s, e, w;
  for (j = 0; j < NY; j++) {
    for (i = 0; i < NX; i++) {

        o =  i + NX*j ; // node( j,i )     n
	n = i+NX*(j+1); // node(j+1,i)     |
	s = i+NX*(j-1); // node(j-1,i)  w--o--e
	e = (i+1)+NX*j; // node(j,i+1)     |
	w = (i-1)+NX*j; // node(j,i-1)     s

	// only update "interior" nodes
	if(i>0 && i<NX-1 && j>0 && j<NY-1) {
	  un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]);
	} else {
	  un[o] = u[o];
	}
    }
  } 
}


/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/
void Set_IC(float * __restrict u0){
  int i, j, o, IC; 

  // select IC
  IC=2;

  switch (IC) {
  case 1: {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
	// set all domain's cells equal to zero
	o = i+NX*j;  u0[o] = 0.0;
	// set BCs in the domain 
	if (j==0)    u0[o] = 0.0; // bottom
	if (i==0)    u0[o] = 0.0; // left
	if (j==NY-1) u0[o] = 1.0; // top
	if (i==NX-1) u0[o] = 1.0; // right
      }
    }
    break;
  }
  case 2: {
    float u_bl = 0.7f;
    float u_br = 1.0f;
    float u_tl = 0.7f;
    float u_tr = 1.0f;

    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
	// set all domain's cells equal to zero
	o = i+NX*j;  u0[o] = 0.0;
	// set BCs in the domain 
	if (j==0)    u0[o] = u_bl + (u_br-u_bl)*i/(NX+1); // bottom
	if (j==NY-1) u0[o] = u_tl + (u_tr-u_tl)*i/(NX+1); // top
	if (i==0)    u0[o] = u_bl + (u_tl-u_bl)*j/(NY+1); // left
	if (i==NX-1) u0[o] = u_br + (u_tr-u_br)*j/(NY+1); // right
      }
    }
    break;
  }
  case 3: {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
	// set all domain's cells equal to zero
	o = i+NX*j;  u0[o] = 0.0;
	// set left wall to 1
	if (i==NX-1) u0[o] = 1.0;
      }
    }
    break;
  }
    // here to add another IC
  }
}

/********/
/* MAIN */
/********/
int main() {

  // --- host temperature distributions
  float *h_T	 = (float *)calloc(NX*NY,sizeof(float));
  float *h_T_old = (float *)calloc(NX*NY,sizeof(float));
  float *h_T_GPU_tex_result= (float *)malloc(NX*NY*sizeof(float));

  // --- Set initial condition
  Set_IC(h_T);
  Set_IC(h_T_old);

  // --- device temperature distribution
  float *d_T_tex;	cudaMalloc((void**)&d_T_tex,	NX*NY*sizeof(float));
  float *d_T_old_tex;	cudaMalloc((void**)&d_T_old_tex,NX*NY*sizeof(float));

  cudaMemcpy(d_T_tex,	h_T,	 NX*NY*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_T_old_tex,d_T_tex,NX*NY*sizeof(float), cudaMemcpyDeviceToDevice);

  /*********************************************/
  /* JACOBI ITERATION FUNCTION - GPU - TEXTURE */
  /*********************************************/
  
  // --- Configure and Bind Textures to global memory
  //cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  cudaBindTexture2D(NULL, &tex_T,	  d_T_tex, &desc,NX,NY,NX*sizeof(float));
  cudaBindTexture2D(NULL, &tex_T_old, d_T_old_tex, &desc,NX,NY,NX*sizeof(float));

  tex_T.addressMode[0] = cudaAddressModeWrap;
  tex_T.addressMode[1] = cudaAddressModeWrap;
  tex_T.filterMode = cudaFilterModePoint;
  tex_T.normalized = false;
	
  tex_T_old.addressMode[0] = cudaAddressModeWrap;
  tex_T_old.addressMode[1] = cudaAddressModeWrap;
  tex_T_old.filterMode = cudaFilterModePoint;
  tex_T_old.normalized = false;

  // --- Grid size
  dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 dimGrid (DIVIDE_INTO(NX, BLOCK_SIZE_X), DIVIDE_INTO(NY, BLOCK_SIZE_Y));

  // Request computer current time
  time_t t = clock();

  // --- Jacobi iterations on the device - texture case
  printf("Using GPU-Texture solver\n");
  for (int step=0; step < NO_STEPS; step+=2) {
    if (step%10000==0) printf("Step %d of %d\n",step,(int)NO_STEPS);
    Laplace2d_texture<<<dimGrid, dimBlock>>>(d_T_old_tex,0); //cudaDeviceSynchronize();
    Laplace2d_texture<<<dimGrid, dimBlock>>>( d_T_tex , 1 ); //cudaDeviceSynchronize();
  }	
    // --- Copy results from device to host
  cudaMemcpy(h_T_GPU_tex_result,d_T_tex,NX*NY*sizeof(float),cudaMemcpyDeviceToHost);

  // Measure and Report computation time
  t = clock()-t; printf("Computing time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

  // --- Unbind textures
  cudaUnbindTexture(tex_T);
  cudaUnbindTexture(tex_T_old);

  /***********************************/
  /* JACOBI ITERATION FUNCTION - CPU */
  /***********************************/
  if (COMPARE) {
  // Request computer current time
  t = clock();

  // --- Jacobi iterations on the host
  printf("Using CPU solver\n");
  for (int step=0; step < NO_STEPS; step+=2) {
    if (step%10000==0) printf("Step %d of %d\n",step,(int)NO_STEPS);
    Laplace2d(h_T,h_T_old); 
    Laplace2d(h_T_old,h_T);
  }

  // Measure and Report computation time
  t = clock()-t; printf("Computing time (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);
  }

  /*******************/
  /* POST-PROCESSING */
  /*******************/
  if (COMPARE) {
  // --- Calculate percentage root mean square error between host and device results
  float sum_tex = 0.f, sum_ref = 0.f;
  for (int j=0; j<NY; j++)
    for (int i=0; i<NX; i++) {
      sum_tex = sum_tex+(h_T_GPU_tex_result[j*NX+i]-
			 h_T[j*NX+i])*(h_T_GPU_tex_result[j*NX+i]-
				       h_T[j*NX+i]);
      sum_ref = sum_ref + h_T[j*NX+i]*h_T[j*NX+i];
    }
  printf("Percentage root mean square error texture   = %f\n", 100.*sqrt(sum_tex / sum_ref));
  }
	
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int j = 0; j < NY; j++) {
      for (int i = 0; i < NX; i++) {
	fprintf(pFile, "%d\t %d\t %g\n",j,i,h_T_GPU_tex_result[i+NX*j]);
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }

  // end program
  return 0;
}
