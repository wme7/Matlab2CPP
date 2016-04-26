
#include "heat3d.h"

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
  // allocate sub-domain for a one-dimensional domain decomposition in the Z-direction
  dmn domain;
  domain.gpu = gpu;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.nx = NX/1;
  domain.ny = NY/1;
  domain.nz = NZ/SZ;
  domain.size = domain.nx*domain.ny*domain.nz;
  domain.rx = 0;
  domain.ry = 0;
  domain.rz = rank;
  
  cudaSetDevice(domain.gpu); 

  // Have process 0 print out some information.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n\n" );
    printf ("  C++/MPI version\n" );
    printf ("  Solve the 3D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf ("  Commence Simulation: procs rank %d out of %d cores with GPU(%d)"
	  " working with (%d +0) x (%d +0) x (%d +2) cells\n",
	  rank,npcs,gpu,domain.nx,domain.ny,domain.nz);

  // return the domain structure
  return domain;
}

void Manage_Memory(int phase, dmn domain, real **h_u, real **t_u, real **d_u, real **d_un){
  size_t global = NX*NY*NZ*sizeof(real);
  size_t local = (domain.nx+0*R)*(domain.ny+0*R)*(domain.nz+2*R)*sizeof(real);
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
  int i, j, k, o; const int XY=NX*NY;
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

void Save_Results(real *u){
  // print result to txt file
  FILE *pFile = fopen("result.txt", "w");  
  const int XY=NX*NY;
  if (pFile != NULL) {
    for (int k = 0;k < NZ; k++) {
      for (int j = 0; j < NY; j++) {
	for (int i = 0; i < NX; i++) {      
	  fprintf(pFile, "%d\t %d\t %d\t %g\n",k,j,i,u[i+NX*j+XY*k]);
	}
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

// void Set_NeumannBC(real *u, const int l, const char letter){
//   int XY=NX*NY, i, j;
//   switch (letter) { 
//   case 'B': { 
//     for (j = 0; j < NY; j++) {
//       for (i = 0; i < NX; i++) {
// 	u[i+NX*j+XY*l-1]=u[i+NX*j+XY*l];
//       }
//     }
//     break;
//   }
//   case 'T': { 
//     for (j = 0; j < NY; j++) {
//       for (i = 0; i < NX; i++) {
// 	u[i+NX*j+XY*l+1]=u[i+NX*j+XY*l];
//       }
//     }
//     break;
//   }
//   }
// }

void Manage_Comms(int phase, dmn domain, real **t_u, real **d_u) {
  cudaError_t Error;
  if (phase==0) {
    // Send local domains to their associated GPU
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d) :::\n",phase);
    Error=cudaMemcpy(*d_u,*t_u,(domain.nx+0*R)*(domain.ny+0*R)*(domain.nz+2*R)*sizeof(real),cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (Memcpy h -> d) = %s \n",cudaGetErrorString(Error));
  }
  if (phase==1) {
    // Communicate halo regions
    const int nm= domain.nx*domain.ny; // n*m layer
    const int l = domain.nz;
    MPI_Status status; 
    MPI_Request rqSendUp, rqSendDown, rqRecvUp, rqRecvDown;
  
    // Impose BCs!
    //if (r==  0 ) Set_NeumannBC(*u,1,'B'); // impose Dirichlet BC u[row  1 ]
    //if (r==SZ-1) Set_NeumannBC(*u,l,'T'); // impose Dirichlet BC u[row L-1]

    // Communicate halo regions
    if (domain.rz <SZ-1) {
      MPI_Isend(*d_u+nm*l    ,nm,MPI_CUSTOM_REAL,domain.rz+1,1,MPI_COMM_WORLD,&rqSendDown); // send u[layerL-1] to   rank+1
      MPI_Irecv(*d_u+nm*(l+R),nm,MPI_CUSTOM_REAL,domain.rz+1,0,MPI_COMM_WORLD,&rqRecvUp  ); // recv u[layer L ] from rank+1
    }
    if (domain.rz > 0 ) {
      MPI_Isend(*d_u+nm      ,nm,MPI_CUSTOM_REAL,domain.rz-1,0,MPI_COMM_WORLD,&rqSendUp  ); // send u[layer 1 ] to   rank-1
      MPI_Irecv(*d_u         ,nm,MPI_CUSTOM_REAL,domain.rz-1,1,MPI_COMM_WORLD,&rqRecvDown); // recv u[layer 0 ] from rank-1
    }

    // Wait for process to complete
    if(domain.rz <SZ-1) {
      MPI_Wait(&rqSendDown, &status);
      MPI_Wait(&rqRecvUp,   &status);
    }
    if(domain.rz > 0 ) {
      MPI_Wait(&rqRecvDown, &status);
      MPI_Wait(&rqSendUp,   &status);
    }
  }
  if (phase==2) {
    // Collect local domains from their associated GPU
    if (DEBUG) printf("::: Perform GPU-CPU comms (phase %d) :::\n",phase);
    Error=cudaMemcpy(*t_u,*d_u,(domain.nx+0*R)*(domain.ny+0*R)*(domain.nz+2*R)*sizeof(real),cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (Memcpy d -> h) = %s \n",cudaGetErrorString(Error));
  }
}

__global__ void Laplace3d(const int nz, const int rz, const real * __restrict__ u, real * __restrict__ un){
  int o, n, s, e, w, t, b;  
  const int XY=NX*NY;
  // Threads id
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  const int j = threadIdx.y + blockIdx.y*blockDim.y;
  const int k = threadIdx.z + blockIdx.z*blockDim.z;

  o = i+(NX*j)+(XY*k); // node( j,i,k )      n  b
  n = o + NX;          // node(j+1,i,k)      | /
  s = o - NX;          // node(j-1,i,k)      |/
  e = o + 1;           // node(j,i+1,k)  w---o---e
  w = o - 1;           // node(j,i-1,k)     /|
  t = o + XY;          // node(j,i,k+1)    / |
  b = o - XY;          // node(j,i,k-1)   t  s

  // only update "interior" nodes
  if(i>0 && i<NX-1 && j>0 && j<NY-1 && k>0 && k<nz-1) {
    un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]) + KZ*(u[t]-2*u[o]+u[b]);
  } else {
    un[o] = u[o];
  }
}

void Call_Laplace(dmn domain, real **u, real **un) {
  // Produce one iteration of the laplace operator
  //cudaSetDevice(domain.gpu); 
  dim3 dimBlock=dim3(8,8,8);
  dim3 dimGrid =dim3((domain.nx+8-1)/8,(domain.ny+8-1)/8,(domain.nz+8-1)/8); 
  Laplace3d<<<dimGrid,dimBlock>>>(domain.ny+2*R,domain.ry,*u,*un);
  if (DEBUG) printf("CUDA error (Laplace3d) %s\n",cudaGetErrorString(cudaPeekAtLastError()));
  cudaError_t Error = cudaDeviceSynchronize();
  if (DEBUG) printf("CUDA error (Laplace3d Synchronize) %s\n",cudaGetErrorString(Error));
}

// void Laplace2d(const int nz, const real * __restrict__ u, real * __restrict__ un){
//   // Using (i,j,k) = [i+N*j+M*N*k] indexes
//   int i, j, k, o, n, s, e, w, t, b; 
//   const int XY=NX*NY;
//   for (k = 0; k < nz; k++) {
//     for (j = 0; j < NY; j++) {
//       for (i = 0; i < NX; i++) {
	
// 	o = i+ (NX*j) + (XY*k); // node( j,i,k )      n  b
// 	n = (i==NX-1) ? o:o+NX; // node(j+1,i,k)      | /
// 	s = (i==0)    ? o:o-NX; // node(j-1,i,k)      |/
// 	e = (j==NY-1) ? o:o+1;  // node(j,i+1,k)  w---o---e
// 	w = (j==0)    ? o:o-1;  // node(j,i-1,k)     /|
// 	t =               o+XY; // node(j,i,k+1)    / |
// 	b =               o-XY; // node(j,i,k-1)   t  s

// 	// only update "interior" nodes
// 	if (k>0 && k<nz-1) {
// 	  un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]) + KZ*(u[t]-2*u[o]+u[b]);
// 	} else {
// 	  un[o] = u[o];
// 	}
//       }
//     } 
//   }
// }

// void Call_Laplace(dmn domain, real **u, real **un){
//   // Produce one iteration of the laplace operator
//   Laplace2d(domain.nz+2*R,*u,*un);
// }
