
#include "heat2d.h"

dmn Manage_Domain(int rank, int npcs, int *coord, int *ngbr){
  // allocate sub-domain for a one-dimensional domain decomposition in the Y-direction
  dmn domain;
  domain.rank = rank;
  domain.npcs = npcs;
  domain.nx = NX/1;  /* NX/SX */
  domain.ny = NY/SY;
  domain.size = domain.nx*domain.ny;
  domain.rx = coord[1];
  domain.ry = coord[0];
  domain.u = ngbr[UP];
  domain.d = ngbr[DOWN];
  
  // Have process 0 print out some information.
  if (rank==ROOT) {
    printf ("HEAT_MPI:\n\n" );
    printf ("  C++/MPI version\n" );
    printf ("  Solve the 2D time-dependent heat equation.\n\n" );
  } 

  // Print welcome message
  printf ("  Commence Simulation: procs rank %d out of %d cores"
	  " working with (%d +0) x (%d +%d) cells\n",rank,npcs,domain.nx,domain.ny,2*R);

  return domain;
}

void Manage_Memory(int phase, dmn domain, real **g_u, real **t_u, real **t_un){
  if (phase==0) {
    // Allocate global domain on ROOT
    if (domain.rank==ROOT) *g_u=(real*)malloc(NX*NY*sizeof(real)); // only exist in ROOT!
    // Allocate local domains on MPI threats with 2 extra slots for halo regions
    *t_u =(real*)malloc((domain.nx+0*R)*(domain.ny+2*R)*sizeof(real));
    *t_un=(real*)malloc((domain.nx+0*R)*(domain.ny+2*R)*sizeof(real));
    memset(*t_u ,0,(domain.nx+0*R)*(domain.ny+2*R));
    memset(*t_un,0,(domain.nx+0*R)*(domain.ny+2*R));
  }
  if (phase==1) {
    // Free the domain on host
    if (domain.rank==ROOT) free(*g_u);
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
  switch (letter) {
  case 'B': { /* bottom BC */
    float u_bl = 0.7f;
    float u_br = 1.0f;
    int j = R;
    for (int i = 0; i < NX; i++) u[i+NX*j] = u_bl + (u_br-u_bl)*i/(NX-1); break;
  }
  case 'T': { /* top BC */
    float u_tl = 0.7f;
    float u_tr = 1.0f;
    int j = domain.ny;
    for (int i = 0; i < NX; i++) u[i+NX*j] = u_tl + (u_tr-u_tl)*i/(NX-1); break;
  }
  }
}

void Set_NeumannBC(real *u, const int j, const char letter){
  switch (letter) {
  case 'B': { /* u[ 1 ]=u[ 2 ] */; break;}
  case 'T': { /* u[ n ]=u[n-1] */; break;}
  }
}

void Manage_Comms(dmn domain, MPI_Comm Comm2d, real *u) {
  const int nx = domain.nx;
  const int ny = domain.ny;
  
// Impose BCs at the top and bottom regions
  if (domain.ry==  0 ) Set_DirichletBC(domain, u,'B'); // impose Dirichlet BC u[row  1 ]
  if (domain.ry==SY-1) Set_DirichletBC(domain, u,'T'); // impose Dirichlet BC u[row M-1]

  // Exchage Halo regions:  top and bottom neighbors 
  MPI_Sendrecv( &(u[ nx*  ny  ]), nx, MPI_CUSTOM_REAL, domain.u, 1, 
                &(u[ nx*   0  ]), nx, MPI_CUSTOM_REAL, domain.d, 1, 
                Comm2d, MPI_STATUS_IGNORE);
  MPI_Sendrecv( &(u[ nx*  R   ]), nx, MPI_CUSTOM_REAL, domain.d, 2, 
                &(u[ nx*(ny+R)]), nx, MPI_CUSTOM_REAL, domain.u, 2, 
                Comm2d, MPI_STATUS_IGNORE);
}

void Laplace2d(const int nx, const int ny, const real * __restrict__ u, real * __restrict__ un){
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
  Laplace2d(domain.nx+0*R,domain.ny+2*R,*u,*un);
}