
#include "heat2d.h"

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
