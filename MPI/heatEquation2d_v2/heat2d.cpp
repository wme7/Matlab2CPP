
#include "heat2d.h"

dmn Manage_Domain(int rank, int npcs, int *coord, int *ngbr){
  // allocate sub-domain for a one-dimensional domain decomposition in the Y-direction
  dmn domain;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.ny = NY/SY;
  domain.nx = NX/SX;
  domain.size = domain.nx*domain.ny;
  domain.rx = coord[0];
  domain.ry = coord[1];
  domain.u = ngbr[UP];
  domain.d = ngbr[DOWN];
  domain.l = ngbr[LEFT];
  domain.r = ngbr[RIGHT];
  
  // Have process 0 print out some information.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n\n" );
    printf ("  C++/MPI version\n" );
    printf ("  Solve the 2D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf ("  Commence Simulation: procs rank %d out of %d cores"
	  " working with (%d +%d) x (%d +%d) cells\n",rank,npcs,domain.nx,2*R,domain.ny,2*R);

  return domain;
}

void Manage_DataTypes(int phase, dmn domain, 
		      MPI_Datatype *xSlice, MPI_Datatype *ySlice, 
		      MPI_Datatype *myGlobal, MPI_Datatype *myLocal){
  MPI_Datatype global;
  int nx = domain.nx;
  int ny = domain.ny;

  if (phase==0) {
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
    MPI_Type_commit(ySlice);
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

void Set_DirichletBC(double *u, const int j, const char letter){
  switch (letter) {
  case 'B': { /* bottom BC */
    float u_bl = 0.7f;
    float u_br = 1.0f;
    for (int i = 0; i < NX; i++) u[i+NX*j] = u_bl + (u_br-u_bl)*i/(NX-1); break;
  }
  case 'T': { /* top BC */
    float u_tl = 0.7f;
    float u_tr = 1.0f;
    for (int i = 0; i < NX; i++) u[i+NX*j] = u_tl + (u_tr-u_tl)*i/(NX-1); break;
  }
  }
}

void Set_NeumannBC(double *u, const int j, const char letter){
  switch (letter) {
  case 'B': { /* u[ 1 ]=u[ 2 ] */; break;}
  case 'T': { /* u[ n ]=u[n-1] */; break;}
  }
}

void Manage_Comms(dmn domain, double **u) {
  const int nx = domain.nx;
  const int ny = domain.ny;
  
  // Impose BCs!
  if (r== 0 ) Set_DirichletBC(*u,1,'B'); // impose Dirichlet BC u[row  1 ]
  if (r==p-1) Set_DirichletBC(*u,m,'T'); // impose Dirichlet BC u[row M-1]

  // Exchage Halo regions

  // Exchange x - slices with top and bottom neighbors 
  MPI_Sendrecv(&(t_u[  ny  *(nx+2*R)+1]), 1, xSlice, nbrs[UP]  , 1, 
	       &(t_u[  0   *(nx+2*R)+1]), 1, xSlice, nbrs[DOWN], 1, 
	       Comm2d, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(t_u[  1   *(nx+2*R)+1]), 1, xSlice, nbrs[DOWN], 2, 
	       &(t_u[(ny+1)*(nx+2*R)+1]), 1, xSlice, nbrs[UP]  , 2, 
	       Comm2d, MPI_STATUS_IGNORE);
  // Exchange y - slices with left and right neighbors 
  MPI_Sendrecv(&(t_u[1*(nx+2*R)+  nx  ]), 1, ySlice, nbrs[RIGHT],3, 
	       &(t_u[1*(nx+2*R)+   0  ]), 1, ySlice, nbrs[LEFT] ,3, 
	       Comm2d, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(t_u[1*(nx+2*R)+   1  ]), 1, ySlice, nbrs[LEFT] ,4, 
	       &(t_u[1*(nx+2*R)+(nx+1)]), 1, ySlice, nbrs[RIGHT],4, 
	       Comm2d, MPI_STATUS_IGNORE);
}

void Laplace2d(const int ny, const double * __restrict__ u, double * __restrict__ un){
  // Using (i,j) = [i+N*j] indexes
  int o, n, s, e, w;
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < NX; i++) {

      o =  i + NX*j ; // node( j,i )     n
      n = i+NX*(j+1); // node(j+1,i)     |
      s = i+NX*(j-1); // node(j-1,i)  w--o--e
      e = (i+1)+NX*j; // node(j,i+1)     |
      w = (i-1)+NX*j; // node(j,i-1)     s
      
      // only update "interior" nodes
      if(i>0 && i<NX-1 && j>0 && j<ny-1) {
	un[o] = u[o] + KX*(u[e]-2*u[o]+u[w]) + KY*(u[n]-2*u[o]+u[s]);
      } else {
	un[o] = u[o];
      }
    }
  } 
}

void Call_Laplace(dmn domain, double **u, double **un){
  // Produce one iteration of the laplace operator
  Laplace2d(domain.ny+2*R,*u,*un);
}
