
#include "heat3d.h"

dmn Manage_Domain(int rank, int npcs, int *coord, int *ngbr){
  // allocate sub-domain for a one-dimensional domain decomposition in the Y-direction
  dmn domain;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.nx = NX/1; /* NX/SX */;
  domain.ny = NY/1; /* NY/SY */;
  domain.nz = NZ/SZ;
  domain.size = domain.nx*domain.ny*domain.nz;
  domain.rx = coord[2];
  domain.ry = coord[1];
  domain.rz = coord[0];
  domain.t = ngbr[TOP];
  domain.b = ngbr[BOTTOM];
  domain.n = ngbr[NORTH];
  domain.s = ngbr[SOUTH];
  domain.e = ngbr[EAST];
  domain.w = ngbr[WEST];
  
  // Have process 0 print out some information.
  if (rank==ROOT) {
    printf("HEAT_MPI:\n\n" );
    printf("  C++/MPI version\n" );
    printf("  Solve the 2D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf("  Commence Simulation:");
  printf("  procs rank %2d (ry=%d,rx=%d,rz=%d) out of %2d cores"
	 " working with (%d +%d) x (%d +%d) x (%d +%d) cells\n",
	 rank,domain.ry,domain.rx,domain.rz,npcs,domain.nx,2*R,domain.ny,2*R,domain.nz,2*R);

  return domain;
}

void Manage_DataTypes(int phase, dmn domain, 
		      MPI_Datatype *xySlice, MPI_Datatype *yzSlice, MPI_Datatype *xzSlice,
		      MPI_Datatype *myGlobal, MPI_Datatype *myLocal){
  if (phase==0) { /*
    MPI_Datatype global;
    int nx = domain.nx;
    int ny = domain.ny;
    int nz = domain.nz;

    // Build a MPI data type for a subarray in Root processor
    int bigsizes[3] = {NZ,NY,NX};
    int subsizes[3] = {nz,ny,nx};
    int starts[3] = {0,0,0};
    MPI_Type_create_subarray(3, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_CUSTOM_REAL, &global);
    MPI_Type_create_resized(global, 0, nx*sizeof(real), myGlobal); // extend the type 
    MPI_Type_commit(myGlobal);
    
    // Build a MPI data type for a subarray in workers
    int bigsizes2[3] = {R+nz+R,R+ny+R,R+nx+R};
    int subsizes2[3] = {nz,ny,nx};
    int starts2[3] = {R,R,R};
    MPI_Type_create_subarray(3, bigsizes2, subsizes2, starts2, MPI_ORDER_C, MPI_CUSTOM_REAL, myLocal);
    MPI_Type_commit(myLocal); // now we can use this MPI costum data type

    // Halo data types
    MPI_Type_vector(nx, 1,    1  , MPI_CUSTOM_REAL, xSlice);
    MPI_Type_vector(ny, 1, R+nx+R, MPI_CUSTOM_REAL, ySlice);
    MPI_Type_commit(xSlice);
    MPI_Type_commit(ySlice); */
  }
  if (phase==1) {
    MPI_Type_free(xySlice);
    MPI_Type_free(xzSlice);
    MPI_Type_free(yzSlice);
    MPI_Type_free(myLocal);
    MPI_Type_free(myGlobal);
  }
}

void Manage_Memory(int phase, dmn domain, real **h_u, real **t_u, real **t_un){
  if (phase==0) {
    // Allocate global domain on ROOT
    if (domain.rank==ROOT) *h_u=(real*)malloc(NX*NY*NZ*sizeof(real)); // only exist in ROOT!
    // Allocate local domains on MPI threats with 2 extra slots for halo regions
    *t_u =(real*)malloc((domain.nx+2*R)*(domain.ny+2*R)*(domain.nz+2*R)*sizeof(real)); 
    *t_un=(real*)malloc((domain.nx+2*R)*(domain.ny+2*R)*(domain.nz+2*R)*sizeof(real));
    memset(*t_u ,0,(domain.nx+2*R)*(domain.ny+2*R)*(domain.nz+2*R));
    memset(*t_un,0,(domain.nx+2*R)*(domain.ny+2*R)*(domain.nz+2*R));
  }
  if (phase==1) {
    // Free the domain on host
    if (domain.rank==ROOT) free(*h_u);
    free(*t_u);
    free(*t_un);
  }
}

/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/
void Call_IC(const int IC, real * __restrict u0){
  int i, j, k, o; const int XY=NX*NY;
  switch (IC) {
  case 0: { /* for verification only! */
    for (k = 0; k < NZ; k++) {
      for (j = 0; j < NY; j++) {
	for (i = 0; i < NX; i++) {
	  // set all domain's cells equal to 0.1
	  o = i+NX*j+XY*k;  u0[o] = o;
	}
      }
    }
    break;
  }
  case 1: {
    for (k = 0; k < NZ; k++) {
      for (j = 0; j < NY; j++) {
	for (i = 0; i < NX; i++) {
	  // set all domain's cells equal to 0.1
	  o = i+NX*j+XY*k;  u0[o] = 0.1;
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

void Print(real *data, int nx, int ny, int nz) {    
  printf("-- Memory --\n"); int xy=nx*ny;
  for (int k=0; k<nz; k++) {
    printf("-- layer %d --\n",k);
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
	printf("%3.0f ", data[i+nx*j+xy*k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}


void Set_NeumannBC(dmn domain, real *u, const char letter){
  // corrections for global indexes
  int i, j, k, n=domain.nx+2*R, m=(domain.nx+2*R)*(domain.ny+2*R); // lengths 

  switch (letter) {
  case 'W': { /* west BC */
    i = R;
    for (k=0; k<domain.nz; k++) 
      for (j=0; j<domain.ny; j++) 
	u[(i-1)+n*(j+R)+m*(k+R)] = u[i+n*(j+R)+m*(k+R)]; 
  }
  case 'E': { /* east BC */
    i = domain.nx;
    for (k=0; k<domain.nz; k++) 
      for (j=0; j<domain.ny; j++) 
	u[(i+1)+n*(j+R)+m*(k+R)] = u[i+n*(j+R)+m*(k+R)]; 
  }
  case 'S': { /* south BC */
    j = R;
    for (k=0; k<domain.nz; k++) 
      for (i=0; i<domain.nx; i++) 
	u[(i+R)+n*(j-1)+m*(k+R)] = u[(i+R)+n*j+m*(k+R)]; 
  }
  case 'N': { /* north BC */
    j = domain.ny;
    for (k=0; k<domain.nz; k++) 
      for (i=0; i<domain.nx; i++) 
	u[(i+R)+n*(j+1)+m*(k+R)] = u[(i+R)+n*j+m*(k+R)]; 
  }
  case 'B': { /* bottom BC */
    k = R;
    for (j=0; j<domain.ny; j++) 
      for (i=0; i<domain.nx; i++) 
	u[(i+R)+n*(j+R)+m*(k-1)] = u[(i+R)+n*(j+R)+m*k]; 
  }
  case 'T': { /* top BC */
    k = domain.nz;
    for (j=0; j<domain.ny; j++) 
      for (i=0; i<domain.nx; i++) 
	u[(i+R)+n*(j+R)+m*(k+1)] = u[(i+R)+n*(j+R)+m*k]; 
  }
  }
}

void Manage_Comms(dmn domain, MPI_Comm Comm3d, MPI_Datatype xySlice, MPI_Datatype yzSlice, MPI_Datatype xzSlice, real *u) {
  const int n = R+domain.nx+R, m=(domain.nx+2*R)*(domain.ny+2*R); // lengths 
  
  // Impose BCs!
  if (domain.rx==  0 ) Set_NeumannBC(domain, u,'W'); 
  if (domain.rx==SX-1) Set_NeumannBC(domain, u,'E'); 
  if (domain.ry==  0 ) Set_NeumannBC(domain, u,'S'); 
  if (domain.ry==SY-1) Set_NeumannBC(domain, u,'N'); 
  if (domain.rz==  0 ) Set_NeumannBC(domain, u,'B'); 
  if (domain.rz==SZ-1) Set_NeumannBC(domain, u,'T'); 
    
    // Exchange xy - slices with top and bottom neighbors 
    MPI_Sendrecv(&(u[ R+ n*R+ m*domain.nz ]), 1, xySlice, domain.b,1, 
		 &(u[    R+ n*R+ m*0      ]), 1, xySlice, domain.t,1, 
		 Comm3d, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&(u[    R+ n*R+ m*R      ]), 1, xySlice, domain.t,2, 
		 &(u[R+n*R+m*(domain.nz+1)]), 1, xySlice, domain.b,2, 
		 Comm3d, MPI_STATUS_IGNORE);
    // Exchange yz - slices with left and right neighbors 
    MPI_Sendrecv(&(u[ domain.nx+ n*R+ m*R ]), 1, yzSlice, domain.e,3, 
		 &(u[    0 + n*R+ m*R     ]), 1, yzSlice, domain.w,3, 
		 Comm3d, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&(u[    R + n*R+ m*R     ]), 1, yzSlice, domain.w,4, 
    		 &(u[(domain.nx+1)+n*R+m*R]), 1, yzSlice, domain.e,4, 
		 Comm3d, MPI_STATUS_IGNORE);
    // Exchange xz - slices with south and north neighbors 
    MPI_Sendrecv(&(u[ R+ n*domain.ny+ m*R ]), 1, xzSlice, domain.s,5, 
		 &(u[    R+ n*0 + m*R     ]), 1, xzSlice, domain.n,5, 
		 Comm3d, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&(u[    R+ n*R + m*R     ]), 1, xzSlice, domain.n,6, 
    		 &(u[R+n*(domain.ny+1)+m*R]), 1, xzSlice, domain.s,6, 
		 Comm3d, MPI_STATUS_IGNORE);
}

void Laplace3d(const int nx, const int ny, const int nz, 
	       const int rx, const int ry, const int rz, 
	       const real * __restrict__ u, real * __restrict__ un){
  // Using (i,j,k) = [i+N*j+M*N*k] indexes
  int i, j, k, o, n, s, e, w, t, b; 
  const int xy=nx*ny;
  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
	
	o = i + nx*j + xy*k; // node( j,i,k )      n  b
	n = i+nx*(j+1)+xy*k; // node(j+1,i,k)      | /
	s = i+nx*(j-1)+xy*k; // node(j-1,i,k)      |/
	e = (i+1)+nx*j+xy*k; // node(j,i+1,k)  w---o---e
	w = (i-1)+nx*j+xy*k; // node(j,i-1,k)     /|
	t = i+nx*j+xy*(k+1); // node(j,i,k+1)    / |
	b = i+nx*j+xy*(k-1); // node(j,i,k-1)   t  s

	// only update "interior" nodes
	if (i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1) {
	  un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]) + KZ*(u[t]-2*u[o]+u[b]);
	} else {
	  un[o] = u[o];
	}
      }
    } 
  }
}

void Call_Laplace(dmn domain, real **u, real **un){
  // Produce one iteration of the laplace operator
  Laplace3d(domain.nx+2*R,domain.ny+2*R,domain.nz+2*R,domain.rx,domain.ry,domain.rz,*u,*un);
}
