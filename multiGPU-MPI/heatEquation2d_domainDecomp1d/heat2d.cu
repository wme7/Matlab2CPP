
#include "heat2d.h"

void Manage_Devices(){
  // we set devices before MPI init to make sure MPI is aware of the cuda devices!
  char * localRankStr = NULL;
  int rank = 0, devCount = 0;
  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv(ENV_LOCAL_RANK)) != NULL) { rank = atoi(localRankStr); }
  cudaGetDeviceCount(&devCount);
  //cudaSetDevice(rank/devCount);
  cudaSetDevice(rank);
  printf("number of devices found: %d\n",devCount);
}

dmn Manage_Domain(int rank, int npcs, int gpu){
  // allocate sub-domain for a one-dimensional domain decomposition in the Y-direction
  dmn domain;
  domain.gpu = gpu;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.nx = NX/1;
  domain.ny = NY/SY;
  domain.size = domain.nx*domain.ny;
  domain.rx = 0;
  domain.ry = rank;

  cudaSetDevice(domain.gpu); 
  
  // Have process 0 print out some information.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n\n" );
    printf ("  C++/MPI version\n" );
    printf ("  Solve the 2D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf ("  Commence Simulation: cpu rank %d out of %d cores with GPU(%d)"
	  " working with (%d +0) x (%d +%d) cells\n",rank,npcs,gpu,domain.nx,domain.ny,2*R);

  // return the domain structure
  return domain;
}

void Manage_Memory(int phase, dmn domain, real **h_u, real **t_u, real **d_u, real **d_un){
  size_t global = NY*NX*sizeof(real);
  size_t local = (domain.nx+0*R)*(domain.ny+2*R)*sizeof(real);
  cudaError_t Error;
  if (phase==0) {
    // Allocate global domain on ROOT
    if (domain.rank==ROOT) *h_u=(real*)malloc(global);
    // Allocate local domains on MPI threats with 2 extra slots for halo regions
    *t_u =(real*)malloc(local);
    // Allocate local domains on devices with 2 extra slots for halo regions
    Error = cudaSetDevice(domain.gpu); if (DEBUG) printf("CUDA error (cudaSetDevice) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_u ,local); if (DEBUG) printf("CUDA error (cudaMalloc d_u) = %s\n",cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_un,local); if (DEBUG) printf("CUDA error (cudaMalloc d_un) = %s\n",cudaGetErrorString(Error));
  }
  if (phase==1){
    // Free local domain variable on device
    //Error = cudaSetDevice(domain.gpu); if (DEBUG) printf("CUDA error (cudaSetDevice) = %s\n",cudaGetErrorString(Error));
    Error = cudaFree(*d_u ); if (DEBUG) printf("CUDA error (cudaFree d_u) = %s\n",cudaGetErrorString(Error));
    Error = cudaFree(*d_un); if (DEBUG) printf("CUDA error (cudaFree d_un) = %s\n",cudaGetErrorString(Error));
    // Free the local domain on host
    free(*t_u);
  }
  if (phase==2) {
    // Free global domain on ROOT
    if (domain.rank==ROOT) free(*h_u);    
  }
}

/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/
void Call_IC(const int IC, real * __restrict u0){
  int i, j, o; 
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
        if (j==0)    u0[o] = u_bl + (u_br-u_bl)*i/(NX-1); // bottom
        if (j==NY-1) u0[o] = u_tl + (u_tr-u_tl)*i/(NX-1); // top
        if (i==0)    u0[o] = u_bl + (u_tl-u_bl)*j/(NY-1); // left
        if (i==NX-1) u0[o] = u_br + (u_tr-u_br)*j/(NY-1); // right
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

void Save_Results(real *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");
  if (pFile != NULL) {
    for (int j = 0; j < NY; j++) {
      for (int i = 0; i < NX; i++) {      
	fprintf(pFile, "%d\t %d\t %g\n",j,i,u[i+NX*j]);
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

void Print_SubDomain(dmn domain, real *u){
  // print result to terminal
  for (int j = 0; j < domain.ny+2*R; j++) {
    for (int i = 0; i < domain.nx+2*0; i++) {      
      printf("%1.2f ",u[i+domain.nx*j]);
    }
    printf("\n");
  }
  printf("\n");
}

void Print_Domain(dmn domain, real *u){
  // print result to terminal
  for (int j = 0; j < NY; j++) {
    for (int i = 0; i < NX; i++) {      
      printf("%1.2f ",u[i+NX*j]);
    }
    printf("\n");
  }
  printf("\n");
}

__global__ void Set_DirichletBC(const int m, const int rank, real * __restrict__ u){
  // Threads id
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i<NX) {
    if (rank==0) { /* bottom BC */
        float u_bl = 0.7f;
        float u_br = 1.0f;
        int j = 1;
        u[i+NX*j] = u_bl + (u_br-u_bl)*i/(NX-1); 
    }
    if (rank==SY-1) { /* top BC */
        float u_tl = 0.7f;
        float u_tr = 1.0f;
        int j = m;
        u[i+NX*j] = u_tl + (u_tr-u_tl)*i/(NX-1); 
    }
  }
}

void Manage_Comms(int phase, dmn domain, real **t_u, real **d_u) {
  cudaError_t Error;
  if (phase==0) {
    // Send local domains to their associated GPU
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d) :::\n",phase);
    Error=cudaMemcpy(*d_u,*t_u,(domain.nx+0*R)*(domain.ny+2*R)*sizeof(real),cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
  }
  if (phase==1) {
    // Communicate halo regions 
    const int n = domain.nx;
    const int m = domain.ny;
    MPI_Status status; 
    MPI_Request rqSendUp, rqSendDown, rqRecvUp, rqRecvDown;
  
    // Impose BCs!
    int blockSize = 256, gridSize = 1+(n-1)/blockSize;
    Set_DirichletBC<<<gridSize,blockSize>>>(m,domain.ry,*d_u); 

    // Communicate halo regions
    if (domain.ry <SY-1) {
      MPI_Isend(*d_u+n*m    ,n,MPI_CUSTOM_REAL,domain.ry+1,1,MPI_COMM_WORLD,&rqSendDown); // send u[rowM-1] to   rank+1
      MPI_Irecv(*d_u+n*(m+R),n,MPI_CUSTOM_REAL,domain.ry+1,0,MPI_COMM_WORLD,&rqRecvUp  ); // recv u[row M ] from rank+1
    }
    if (domain.ry > 0 ) {
      MPI_Isend(*d_u+n      ,n,MPI_CUSTOM_REAL,domain.ry-1,0,MPI_COMM_WORLD,&rqSendUp  ); // send u[row 1 ] to   rank-1
      MPI_Irecv(*d_u        ,n,MPI_CUSTOM_REAL,domain.ry-1,1,MPI_COMM_WORLD,&rqRecvDown); // recv u[row 0 ] from rank-1
    }

    // Wait for process to complete
    if(domain.ry <SY-1) {
      MPI_Wait(&rqSendDown, &status);
      MPI_Wait(&rqRecvUp,   &status);
    }
    if(domain.ry > 0 ) {
      MPI_Wait(&rqRecvDown, &status);
      MPI_Wait(&rqSendUp,   &status);
    }
  }
  if (phase==2) {
    // Collect local domains from their associated GPU
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d) :::\n",phase);
    Error=cudaMemcpy(*t_u,*d_u,(domain.nx+0*R)*(domain.ny+2*R)*sizeof(real),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
  }
}

__global__ void Laplace2d(const int ny, const int ry, const real * __restrict__ u, real * __restrict__ un){
  int o, n, s, e, w; 
  // Threads id
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  const int j = threadIdx.y + blockIdx.y*blockDim.y;

  o = i+(NX*j); // node( j,i )      n
  n = o + NX;   // node(j+1,i)      |
  s = o - NX;   // node(j-1,i)   w--o--e
  e = o + 1;    // node(j,i+1)      |
  w = o - 1;    // node(j,i-1)      s

  // only update "interior" nodes
  if(i>0 && i<NX-1 && j>0 && j<ny-1) {
    un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]);
  } else {
    un[o] = u[o];
  }
}

extern "C" void Call_Laplace(dmn domain, real **u, real **un){
  // Produce one iteration of the laplace operator
  int tx=32, ty=32; // number of threads in x and y directions
  dim3 blockSize(tx,ty); dim3 numBlocks((domain.nx+tx-1)/tx,(domain.ny+ty-1)/ty); 
  Laplace2d<<<numBlocks,blockSize>>>(domain.ny+2*R,domain.ry,*u,*un);
  if (DEBUG) printf("CUDA error (Laplace2d) %s\n",cudaGetErrorString(cudaPeekAtLastError()));
  cudaError_t Error = cudaDeviceSynchronize();
  if (DEBUG) printf("CUDA error (Laplace2d Synchronize) %s\n",cudaGetErrorString(Error));
}
