
#include "heat3d.h"

/* Initialize textures */
// texture<float, 3, cudaReadModeElementType> tex_u;
// texture<float, 3, cudaReadModeElementType> tex_un;

static cudaTextureObject_t tex_u;
static cudaTextureObject_t tex_un;

/***********************/
/* AUXILIARY FUCNTIONS */
/***********************/
void Print2D(float *u)
{
    // print a single property on terminal
  int XY=NX*NY;
  for (int k = 0; k < NZ; k++) {
    for (int j = 0; j < NY; j++) {
      for (int i = 0; i < NX; i++) {
        printf("%8.2f", u[i+NX*j+XY*k]);
      }
      printf("\n");
    }
    printf("\n\n");
  }
}

void Save_Results(float *u){
  // print result to txt file
  float data;
  FILE *pFile = fopen("result.txt", "w");  
  int XY=NX*NY;
  if (pFile != NULL) {
    for (int k = 0; k < NZ; k++) {
      for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
          data = u[i+NX*j+XY*k];
          fprintf(pFile, "%d\t %d\t %d\t %g\n",k,j,i,data);
        }
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/

void Call_IC(float *__restrict u0){
  int i, j, k, o, IC, XY=NX*NY;

  // select IC
  IC=2;

  switch (IC) {
  case 1: {
    for (k = 0; k < NZ; k++) {
      for (j = 0; j < NY; j++) {
      	for (i = 0; i < NX; i++) {
      	  // set all domain's cells equal to zero
      	  o = i+NX*j+XY*k;  u0[o] = 0.0;
      	  // set BCs in the domain 
      	  if (k==0)    u0[o] = 1.0; // bottom
      	  if (k==NZ-1) u0[o] = 1.0; // top
      	}
      }
    }
    break;
  }
  case 2: {
    for (k = 0; k < NZ; k++) {
      for (j = 0; j < NY; j++) {
        for (i = 0; i < NX; i++) {
          // set all domain's cells equal to zero
          o = i+NX*j+XY*k;  
          u0[o] = 1.0*exp(
            -(DX*(i-NX/2))*(DX*(i-NX/2))/1.5
            -(DY*(j-NY/2))*(DY*(j-NY/2))/1.5
            -(DZ*(k-NZ/2))*(DZ*(k-NZ/2))/12);
        }
      }
    }
    break;
  }
    // here to add another IC
  } 
}

/************************************/
/* LAPLACE ITERATION FUNCTION - CPU */
/************************************/

void Laplace3d_CPU(float *u, float *un){
  // Using (i,j) = [i+N*j+M*N*k] indexes
  int i, j, k, o, n, s, e, w, t, b; 
  int XY=NX*NY;

  for (j = 0; j < NY; j++) {
    for (i = 0; i < NX; i++) {
      for (k = 0; k < NZ; k++) {
	
        o = i+ (NX*j) + (XY*k); // node( j,i,k )      n  b
        n = (i==NX-1) ? o:o+NX; // node(j+1,i,k)      | /
        s = (i==0)    ? o:o-NX; // node(j-1,i,k)      |/
        e = (j==NY-1) ? o:o+1;  // node(j,i+1,k)  w---o---e
        w = (j==0)    ? o:o-1;  // node(j,i-1,k)     /|
        t = (k==NZ-1) ? o:o+XY; // node(j,i,k+1)    / |
        b = (k==0)    ? o:o-XY; // node(j,i,k-1)   t  s

        un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]) + KZ*(u[t]-2*u[o]+u[b]);
      }
    } 
  }
}

void CreateTexture()
{
  //Array creation
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray *d_cuArr;
  cudaMalloc3DArray(&d_cuArr, &channelDesc, make_cudaExtent(SizeNoiseTest,SizeNoiseTest,SizeNoiseTest), 0);
  cudaMemcpy3DParms copyParams = {0};

  //Copy Parameters
  copyParams.srcPtr   = make_cudaPitchedPtr(d_NoiseTest, SizeNoiseTest*sizeof(float), SizeNoiseTest, SizeNoiseTest);
  copyParams.dstArray = d_cuArr;
  copyParams.extent   = make_cudaExtent(SizeNoiseTest,SizeNoiseTest,SizeNoiseTest);
  copyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&copyParams);

  //Array creation End
  cudaResourceDesc    texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));
  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array  = d_cuArr;
  cudaTextureDesc     texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));
  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
  texDescr.addressMode[1] = cudaAddressModeClamp;
  texDescr.addressMode[2] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeElementType;
  cudaCreateTextureObject(&tex_u, &texRes, &texDescr, NULL);
  cudaCreateTextureObject(&tex_un,&texRes, &texDescr, NULL);
}


__global__ void Laplace2d_texture(cudaTextureObject_t tex_u, float * __restrict__ un, const bool flag) {

  // Threads id
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;  
  int k = blockIdx.z * blockDim.z + threadIdx.z;  

  float o, n, s, e, w, t, b; 
  if (flag) {
    o = tex2D(tex_u,i, j ,k); // node( j,i,k )      n  b
    n = tex2D(tex_u,i,j+1,k); // node(j+1,i,k)      | /
    s = tex2D(tex_u,i,j-1,k); // node(j-1,i,k)      |/
    e = tex2D(tex_u,i+1,j,k); // node(j,i+1,k)  w---o---e
    w = tex2D(tex_u,i-1,j,k); // node(j,i-1,k)     /|
    t = tex2D(tex_u,i,j,k+1); // node(j,i,k+1)    / |
    b = tex2D(tex_u,i,j,k-1); // node(j,i,k-1)   t  s
  } else {
    o = tex2D(tex_un,i, j ,k); // node( j,i,k )      n  b
    n = tex2D(tex_un,i,j+1,k); // node(j+1,i,k)      | /
    s = tex2D(tex_un,i,j-1,k); // node(j-1,i,k)      |/
    e = tex2D(tex_un,i+1,j,k); // node(j,i+1,k)  w---o---e
    w = tex2D(tex_un,i-1,j,k); // node(j,i-1,k)     /|
    t = tex2D(tex_un,i,j,k+1); // node(j,i,k+1)    / |
    b = tex2D(tex_un,i,j,k-1); // node(j,i,k-1)   t  s
  }

  // --- Only update "interior" (not boundary) node points
  if (i>0 && i<NX-1 && j>0 && j<NY-1 && k>0 && k<NZ-1) 
    un[i+j*NX+k*NY*NZ] = o + KX*(e-2*o+w) + KY*(n-2*o+s) + KZ*(t-2*o+b);
}

void Call_Laplace(dim3 numBlocks, dim3 threadsPerBlock, cudaArray_t *d_u, cudaArray_t *d_un) 
{
  // Configure and Bind Textures to global memory
  // cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); // or
  cudaChannelFormatDesc desc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);

  tex_u.addressMode[0] = cudaAddressModeWrap;
  tex_u.addressMode[1] = cudaAddressModeWrap;
  tex_u.addressMode[2] = cudaAddressModeWrap;
  tex_u.filterMode = cudaFilterModePoint;
  tex_u.normalized = false;
  
  tex_un.addressMode[0] = cudaAddressModeWrap;
  tex_un.addressMode[1] = cudaAddressModeWrap;
  tex_un.addressMode[2] = cudaAddressModeWrap;
  tex_un.filterMode = cudaFilterModePoint;
  tex_un.normalized = false;

  // Bind textures
  cudaBindTextureToArray(tex_u ,d_u );
  cudaBindTextureToArray(tex_un,d_un);

  // Produce one iteration of the laplace operator
  Laplace2d_texture<<<numBlocks,threadsPerBlock>>>(d_u, 0);
  Laplace2d_texture<<<numBlocks,threadsPerBlock>>>(d_un,1);

  // Unbind textures
  cudaUnbindTexture(tex_u);
  cudaUnbindTexture(tex_un);
}
