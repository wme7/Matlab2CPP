#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEBUG 0 // Display all error messages
//#define NX 256 // number of cells in the x-direction
//#define NY 256 // number of cells in the y-direction
#define L 1.0 // domain length
#define W 1.0 // domain width
#define C 1.0 // c, material conductivity. Uniform assumption.
#define TEND 1.0 // tEnd, output time
#define DX (L/256) // dx, cell size
#define DY (W/256) // dy, cell size
#define DT (1/(2*C*(1/DX/DX+1/DY/DY))) // dt, fix time step size
#define KX (C*DT/(DX*DX)) // numerical conductivity
#define KY (C*DT/(DY*DY)) // numerical conductivity
#define NO_STEPS (TEND/DT) // No. of time steps
#define PI 3.1415926535897932f

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define DIVIDE_INTO(x,y) (((x)+(y)-1)/(y)) // define No. of blocks/warps

// Initialize Textures
texture<float, 2, cudaReadModeElementType>  tex_T;
texture<float, 2, cudaReadModeElementType>  tex_T_old;

/*********************************************/
/* JACOBI ITERATION FUNCTION - GPU - TEXTURE */
/*********************************************/
__global__ void Laplace2d_texture(float * __restrict__ T_new, const bool flag, const int NX, const int NY) {
    
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
  if (i>0 && i<NX-1 && j>0 && j<NY-1) T_new[i+j*NX] = o + KX*(e-2*o+w) + KY*(n-2*o+s);
}

/***********************************/
/* JACOBI ITERATION FUNCTION - CPU */
/***********************************/
void Laplace2d(float * __restrict T, float * __restrict T_new, const int NX, const int NY, const int MAX_ITER)
{
  for(int iter=0; iter<MAX_ITER; iter=iter+2)
    {
      // --- Only update "interior" (not boundary) node points
      for(int j=1; j<NY-1; j++) 
	for(int i=1; i<NX-1; i++) {
	  float T_E = T[(i+1) + NX*j];
	  float T_W = T[(i-1) + NX*j];
	  float T_N = T[i + NX*(j+1)];
	  float T_S = T[i + NX*(j-1)];
	  T_new[i+NX*j] = 0.25*(T_E + T_W + T_N + T_S);
	}
      
      for(int j=1; j<NY-1; j++) 
	for(int i=1; i<NX-1; i++) {
	  float T_E = T_new[(i+1) + NX*j];
	  float T_W = T_new[(i-1) + NX*j];
	  float T_N = T_new[i + NX*(j+1)];
	  float T_S = T_new[i + NX*(j-1)];
	  T[i+NX*j] = 0.25*(T_E + T_W + T_N + T_S);
	}
    }
}

/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/
void Initialize(float * __restrict h_T, const int NX, const int NY)
{
    // --- Set left wall to 1
    for(int j=0; j<NY; j++) h_T[j * NX] = 1.0;
}


/********/
/* MAIN */
/********/
int main()
{
  const int NX = 256;		// --- Number of discretization points along the x axis
  const int NY = 256;		// --- Number of discretization points along the y axis
  const int MAX_ITER = 4000;	// --- Number of Jacobi iterations

  // --- CPU temperature distributions
  float *h_T	 = (float *)calloc(NX*NY,sizeof(float));
  float *h_T_old = (float *)calloc(NX*NY,sizeof(float));
  float *h_T_GPU_tex_result= (float *)malloc(NX*NY*sizeof(float));

  // --- set initial condition
  Initialize(h_T,     NX, NY);
  Initialize(h_T_old, NX, NY);

  // --- GPU temperature distribution
  float *d_T;		cudaMalloc((void**)&d_T,	NX*NY*sizeof(float));
  float *d_T_old;	cudaMalloc((void**)&d_T_old,	NX*NY*sizeof(float));
  float *d_T_tex;	cudaMalloc((void**)&d_T_tex,	NX*NY*sizeof(float));
  float *d_T_old_tex;	cudaMalloc((void**)&d_T_old_tex,NX*NY*sizeof(float));

  cudaMemcpy(d_T,	h_T,	 NX*NY*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_T_tex,	h_T,	 NX*NY*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_T_old,	d_T,	 NX*NY*sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_T_old_tex,d_T_tex,NX*NY*sizeof(float), cudaMemcpyDeviceToDevice);
  
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

  // --- Jacobi iterations on the device - texture case
  for (int k=0; k<MAX_ITER; k=k+2) {
    // Update d_T_tex starting from data stored in d_T_old_tex
    Laplace2d_texture<<<dimGrid, dimBlock>>>(d_T_old_tex,0,NX,NY); cudaDeviceSynchronize();
    // Update d_T_old_tex starting from data stored in d_T_tex
    Laplace2d_texture<<<dimGrid, dimBlock>>>(d_T_tex,1,NX,NY); cudaDeviceSynchronize();
  }	

  // --- Unbind textures
  cudaUnbindTexture(tex_T);
  cudaUnbindTexture(tex_T_old);

  // --- Jacobi iterations on the host
  Laplace2d(h_T,h_T_old,NX,NY,MAX_ITER);

  // --- Copy results from device to host
  cudaMemcpy(h_T_GPU_tex_result,d_T_tex,NX*NY*sizeof(float),cudaMemcpyDeviceToHost);
	
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
