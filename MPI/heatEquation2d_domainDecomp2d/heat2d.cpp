
#include "heat2d.h"

dmn Manage_Domain(int rank, int npcs, int *coord, int *ngbr){
  // allocate sub-domain for a one-dimensional domain decomposition in the Y-direction
  dmn domain;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.ny = NY/SY;
  domain.nx = NX/SX;
  domain.size = domain.nx*domain.ny;
  domain.rx = coord[1];
  domain.ry = coord[0];
  domain.u = ngbr[UP];
  domain.d = ngbr[DOWN];
  domain.l = ngbr[LEFT];
  domain.r = ngbr[RIGHT];
  
  // Have process 0 print out some information.
  if (rank==ROOT) {
    printf("HEAT_MPI:\n\n" );
    printf("  C++/MPI version\n" );
    printf("  Solve the 2D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf("  Commence Simulation:");
  printf("  procs rank %2d (ry=%d,rx=%d) out of %2d cores"
	 " working with (%d +%d) x (%d +%d) cells\n",
	 rank,domain.ry,domain.rx,npcs,domain.nx,2*R,domain.ny,2*R);

  return domain;
}

void Manage_DataTypes(int phase, dmn domain, 
		      MPI_Datatype *xSlice, MPI_Datatype *ySlice, 
		      MPI_Datatype *myGlobal, MPI_Datatype *myLocal){
  if (phase==0) { /*
    MPI_Datatype global;
    int nx = domain.nx;
    int ny = domain.ny;

    // Build a MPI data type for a subarray in Root processor
    int bigsizes[2] = {NY,NX};
    int subsizes[2] = {ny,nx};
    int starts[2] = {0,0};
    MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_CUSTOM_REAL, &global);
    MPI_Type_create_resized(global, 0, nx*sizeof(real), myGlobal); // extend the type 
    MPI_Type_commit(myGlobal);
    
    // Build a MPI data type for a subarray in workers
    int bigsizes2[2] = {R+ny+R,R+nx+R};
    int subsizes2[2] = {ny,nx};
    int starts2[2] = {R,R};
    MPI_Type_create_subarray(2, bigsizes2, subsizes2, starts2, MPI_ORDER_C, MPI_CUSTOM_REAL, myLocal);
    MPI_Type_commit(myLocal); // now we can use this MPI costum data type

    // Halo data types
    MPI_Type_vector(nx, 1,    1  , MPI_CUSTOM_REAL, xSlice);
    MPI_Type_vector(ny, 1, R+nx+R, MPI_CUSTOM_REAL, ySlice);
    MPI_Type_commit(xSlice);
    MPI_Type_commit(ySlice); */
  }
  if (phase==1) {
    MPI_Type_free(xSlice);
    MPI_Type_free(ySlice);
    MPI_Type_free(myLocal);
    MPI_Type_free(myGlobal);
  }
}

void Manage_Memory(int phase, dmn domain, real **h_u, real **t_u, real **t_un){
  if (phase==0) {
    // Allocate global domain on ROOT
    if (domain.rank==ROOT) *h_u=(real*)malloc(NX*NY*sizeof(real)); // only exist in ROOT!
    // Allocate local domains on MPI threats with 2 extra slots for halo regions
    *t_u =(real*)malloc((domain.nx+2*R)*(domain.ny+2*R)*sizeof(real)); 
    *t_un=(real*)malloc((domain.nx+2*R)*(domain.ny+2*R)*sizeof(real));
    memset(*t_u ,0,(domain.nx+2*R)*(domain.ny+2*R));
    memset(*t_un,0,(domain.nx+2*R)*(domain.ny+2*R));
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
    real u_bl = 0.7f;
    real u_br = 1.0f;
    real u_tl = 0.7f;
    real u_tr = 1.0f;

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

void Print(real *data, int nx, int ny) {    
  printf("-- Memory --\n");
  for (int i=0; i<ny; i++) {
    for (int j=0; j<nx; j++) {
      printf("%1.2f ", data[i*nx+j]);
    }
    printf("\n");
  }
}


void Set_DirichletBC(dmn domain, real *u, const char letter){
  // corrections for global indexes
  int xo = domain.rx*domain.nx;
  int yo = domain.ry*domain.ny;
  int n = domain.nx+2*R;

  switch (letter) {
  case 'B': { /* bottom BC */
    real u_bl = 0.7;
    real u_br = 1.0;
    int j = R;
    for (int i = 0; i < domain.nx; i++) u[i+R+n*j] = u_bl + (u_br-u_bl)*(i+xo)/(NX-1); break;
  }
  case 'T': { /* top BC */
    real u_tl = 0.7;
    real u_tr = 1.0;
    int j = domain.ny;
    for (int i = 0; i < domain.nx; i++) u[i+R+n*j] = u_tl + (u_tr-u_tl)*(i+xo)/(NX-1); break;
  }
  case 'L': { /* left BC */
    real u_bl = 0.7;
    real u_tl = 0.7;
    int i = R;
    for (int j = 0; j < domain.ny; j++) u[i+n*(j+R)] = u_bl + (u_tl-u_bl)*(j+yo)/(NY-1); break;
  }
  case 'R': { /* right BC */
    real u_br = 1.0;
    real u_tr = 1.0;
    int i = domain.nx;
    for (int j = 0; j < domain.ny; j++) u[i+n*(j+R)] = u_br + (u_tr-u_br)*(j+yo)/(NY-1); break;
  }
  }
}

void Manage_Comms(dmn domain, MPI_Comm Comm2d, MPI_Datatype xSlice, MPI_Datatype ySlice, real *u) {
  const int nx = domain.nx;
  const int ny = domain.ny;
  const int n = R+domain.nx+R;
  
  // Impose BCs!
  if (domain.rx==  0 ) Set_DirichletBC(domain, u,'L'); 
  if (domain.rx==SX-1) Set_DirichletBC(domain, u,'R'); 
  if (domain.ry==  0 ) Set_DirichletBC(domain, u,'B'); 
  if (domain.ry==SY-1) Set_DirichletBC(domain, u,'T');
  
  // Exchage Halo regions:  top and bottom neighbors 
  MPI_Sendrecv(&(u[ R + n*ny ]), 1, xSlice, domain.u, 1, 
	       &(u[ R + n* 0 ]), 1, xSlice, domain.d, 1, 
	       Comm2d, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(u[ R + n*  1   ]), 1, xSlice, domain.d, 2, 
	       &(u[ R + n*(ny+1)]), 1, xSlice, domain.u, 2, 
	       Comm2d, MPI_STATUS_IGNORE);
  // Exchage Halo regions:  left and right neighbors 
  MPI_Sendrecv(&(u[ nx + n*R ]), 1, ySlice, domain.r, 3, 
	       &(u[  0 + n*R ]), 1, ySlice, domain.l, 3, 
	       Comm2d, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(u[  R    + n*R ]), 1, ySlice, domain.l, 4, 
	       &(u[(nx+1) + n*R ]), 1, ySlice, domain.r, 4, 
	       Comm2d, MPI_STATUS_IGNORE);
}

void Laplace2d(const int nx, const int ny, const int rx, const int ry,
	       const real * __restrict__ u, real * __restrict__ un){
  // Using (i,j) = [i+N*j] indexes
  int o, n, s, e, w;
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {

      o =  i + nx*j ; // node( j,i )     n
      n = i+nx*(j+1); // node(j+1,i)     |
      s = i+nx*(j-1); // node(j-1,i)  w--o--e
      e = (i+1)+nx*j; // node(j,i+1)     |
      w = (i-1)+nx*j; // node(j,i-1)     s
      
      // only update "interior" nodes
      if(i>0 && i<nx-1 && j>0 && j<ny-1) {
	un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]);
      } else {
	un[o] = u[o];
      }
    }
  } 
}

void Call_Laplace(dmn domain, real **u, real **un){
  // Produce one iteration of the laplace operator
  Laplace2d(domain.nx+2*R,domain.ny+2*R,domain.rx,domain.ry,*u,*un);
}
