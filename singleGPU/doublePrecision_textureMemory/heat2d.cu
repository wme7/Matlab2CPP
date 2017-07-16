
#include "heat2d.h"

/* Initialize textures */
// texture<float, 2, cudaReadModeElementType> tex_u;
// texture<float, 2, cudaReadModeElementType> tex_un;
texture<int2, 2, cudaReadModeElementType> tex_u;
texture<int2, 2, cudaReadModeElementType> tex_un;

static __inline__ __device__ double fetch_double(int2 p){
    return __hiloint2double(p.y, p.x);
}

/***********************/
/* AUXILIARY FUCNTIONS */
/***********************/
void Print2D(double *u, const unsigned int nx, const unsigned int ny)
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

void Save_Results(double *u){
  // print result to txt file
  float data;
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int j = 0; j < NY; j++) {
      for (int i = 0; i < NX; i++) {      
        data = u[i+NX*j]; 
        fprintf(pFile, "%d\t %d\t %g\n",j,i,data);
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
void Call_IC(double * __restrict u0){
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
    double u_bl = 0.7f;
    double u_br = 1.0f;
    double u_tl = 0.7f;
    double u_tr = 1.0f;

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

__global__ void Laplace2d_texture(double * __restrict__ un, const bool flag) {

  // Threads id
  const int i = blockIdx.x * blockDim.x + threadIdx.x ;
  const int j = blockIdx.y * blockDim.y + threadIdx.y ;  

  double o, n, s, e, w; 
  int2 uData;
  if (flag) {
    uData = tex2D(tex_u,i, j ); o = fetch_double(uData); // node( i,j )     n
    uData = tex2D(tex_u,i,j+1); n = fetch_double(uData); // node(i,j+1)     |
    uData = tex2D(tex_u,i,j-1); s = fetch_double(uData); // node(i,j-1)  w--o--e
    uData = tex2D(tex_u,i+1,j); e = fetch_double(uData); // node(i+1,j)     |
    uData = tex2D(tex_u,i-1,j); w = fetch_double(uData); // node(i-1,j)     s
  } else {
    uData = tex2D(tex_un,i, j ); o = fetch_double(uData); // node( i,j )     n
    uData = tex2D(tex_un,i,j+1); n = fetch_double(uData); // node(i,j+1)     |
    uData = tex2D(tex_un,i,j-1); s = fetch_double(uData); // node(i,j-1)  w--o--e
    uData = tex2D(tex_un,i+1,j); e = fetch_double(uData); // node(i+1,j)     |
    uData = tex2D(tex_un,i-1,j); w = fetch_double(uData); // node(i-1,j)     s
  }

  // float o, n, s, e, w;
  // if (flag) {
  //   o = tex2D(tex_u,i, j ); // node( i,j )     n
  //   n = tex2D(tex_u,i,j+1); // node(i,j+1)     |
  //   s = tex2D(tex_u,i,j-1); // node(i,j-1)  w--o--e
  //   e = tex2D(tex_u,i+1,j); // node(i+1,j)     |
  //   w = tex2D(tex_u,i-1,j); // node(i-1,j)     s
  // } else {
  //   o = tex2D(tex_un,i, j ); // node( i,j )     n
  //   n = tex2D(tex_un,i,j+1); // node(i,j+1)     |
  //   s = tex2D(tex_un,i,j-1); // node(i,j-1)  w--o--e
  //   e = tex2D(tex_un,i+1,j); // node(i+1,j)     |
  //   w = tex2D(tex_un,i-1,j); // node(i-1,j)     s
  // }
  // --- Only update "interior" (not boundary) node points
  if (i>0 && i<NX-1 && j>0 && j<NY-1) un[i+j*NX] = o + KX*(e-2*o+w) + KY*(n-2*o+s);
}

void Call_Laplace(dim3 numBlocks, dim3 threadsPerBlock, double *d_u, double *d_un) 
{
  // Configure and Bind Textures to global memory
  // cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  // cudaChannelFormatDesc desc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
  cudaChannelFormatDesc desc = cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindSigned);

  tex_u.addressMode[0] = cudaAddressModeWrap;
  tex_u.addressMode[1] = cudaAddressModeWrap;
  tex_u.filterMode = cudaFilterModePoint;
  tex_u.normalized = false;
  
  tex_un.addressMode[0] = cudaAddressModeWrap;
  tex_un.addressMode[1] = cudaAddressModeWrap;
  tex_un.filterMode = cudaFilterModePoint;
  tex_un.normalized = false;

  // bind textures
  // cudaBindTexture2D(0,&tex_u, d_u, &desc,NX,NY,NX*sizeof(float));
  // cudaBindTexture2D(0,&tex_un,d_un,&desc,NX,NY,NX*sizeof(float));
  cudaBindTexture2D(0,&tex_u, d_u, &desc,NX,NY,NX*sizeof(double));
  cudaBindTexture2D(0,&tex_un,d_un,&desc,NX,NY,NX*sizeof(double));

  // Produce one iteration of the laplace operator
  Laplace2d_texture<<<numBlocks,threadsPerBlock>>>(d_u, 0);
  Laplace2d_texture<<<numBlocks,threadsPerBlock>>>(d_un,1);

  // Unbind textures
  cudaUnbindTexture(tex_u);
  cudaUnbindTexture(tex_un);
}
