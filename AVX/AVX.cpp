/* Tutorial #2: 3D Transient HT using OpenMP and AVX intrinsic functions
   Written by: Matthew Smith, Department of Mechanical Engineering, NCKU
   For corrections or queries: msmith@mail.ncku.edu.tw

   Function Description
   
   Allocate_Memory():   	Allocate the memory for variables a,b and c.
   Free_Memory():			Free the variables a,b and c from memory.
   Init();					Set the values of a and b.
   Compute_OpenMP();		Compute 3D FTCS using OpenMP parallelization alone
   Compute_OpenMP_AVX();	Compute 3D FTCS using OpenMP together with AVX Intrinsics
   Save_Data(); 			Save data to file using a TECPLOT ASCII format

   Last Updated: 1st Oct, 2015

   This code is subject to the disclaimers found on hshclncku.com 
   Use at your own risk.

   Reference Paper
   J.-Y. Liu, M.R. Smith, F.A. Kuo and J.-S. Wu, Hybrid OpenMP / AVX acceleration of a 
   split HLL Finite Volume Method for the Shallow Water and Euler Equations,
   Computers and Fluids, 110, pp. 181-188, 2015.	

*/	

#include <stdio.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

#define NX 64				// No. of cells in x direction
#define NY 64				// No. of cells in y direction
#define NZ 64				// No. of cells in z direction
#define N (NX*NY*NZ)     	// N = total number of cells in domain
#define USE_INTRINSICS 1	// 1 = Use AVX Intrinsics, 2 = Use standard code
#define L 0.05				// L = length of domain (m)
#define H 0.03				// H = Height of domain (m)
#define W 0.03				// W = Width of domain (m)
#define DX (L/NX)			// DX, DY, DZ = grid spacing in x,y,z.
#define DY (H/NY)
#define DZ (W/NZ)
#define FINS 15				// FINS, PART, DXFIN - used for geometry 
#define PARTS (FINS*2 + 1)	
#define DXFIN (L/PARTS)
#define ALPHA 0.1			// Thermal diffusivity
#define DT 2.5e-7			// Time step (seconds)
#define PHI_X (DT*ALPHA/(DX*DX))  // CFL in x, y and z respectively.
#define PHI_Y (DT*ALPHA/(DY*DY))
#define PHI_Z (DT*ALPHA/(DZ*DZ))
#define NO_STEPS 200000		// Number of transient time steps
#define OMP_THREADS 16	    // Number of OMP threads 

// Create variables - T in cell (i,j,k) and the surrounding 6 cells (7 in total)
float *a, *b, *c, *d, *e, *f, *g;
float *body;   // In this case, 1 for true (update body temp) and 0 for false
int total_cells = 0; 		// A counter for computed cells

// Some commonly used constants as AVX types
__m256 AVX_PHI_X = _mm256_set1_ps(PHI_X);	
__m256 AVX_PHI_Y = _mm256_set1_ps(PHI_Y);
__m256 AVX_PHI_Z = _mm256_set1_ps(PHI_Z);
__m256 AVX_TWO = _mm256_set1_ps(2.0f);

// Declare our functions
void Allocate_Memory();		// Allocate memory function
void Free_Memory();			// Free memory function
void Init();				// Initialization function
void Compute_OpenMP();		// Solve FTCS using OpenMP only (No intrinsics)
void Compute_OpenMP_AVX();	// Solve FTCS using OpenMP + Intrinsics
void Save_Data();			// Save the data to file


int main() {

	struct timeval start,end;   // Variables for timing
	float time;
	Allocate_Memory();
	Init();
	// Report constants
	printf("Phi (x,y,z) = %g, %g, %g\n", PHI_X, PHI_Y, PHI_Z);
	gettimeofday(&start,NULL); 	// Start stopwatch

	if (USE_INTRINSICS == 0) {
		Compute_OpenMP();
	} else {
		Compute_OpenMP_AVX();
	}		

	gettimeofday(&end,NULL);	// Stop stopwatch and compute time
	time=((end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec)/1000000.0);
	printf("Computation Time = %f sec.\n",time);

	Save_Data();
	Free_Memory();
	return 0;
}

void Allocate_Memory() {
	// Allocate memory
	size_t alignment = 32;
	int error;
	error = posix_memalign((void**)&a, alignment, N*sizeof(float));
	error = posix_memalign((void**)&b, alignment, N*sizeof(float));
	error = posix_memalign((void**)&c, alignment, N*sizeof(float));
	error = posix_memalign((void**)&d, alignment, N*sizeof(float));
	error = posix_memalign((void**)&e, alignment, N*sizeof(float));
	error = posix_memalign((void**)&f, alignment, N*sizeof(float));
	error = posix_memalign((void**)&g, alignment, N*sizeof(float));
	error = posix_memalign((void**)&body, alignment, N*sizeof(float));
}

void Free_Memory() {
	// Free the memory holding our variables
	free(a);
	free(b);
	free(c);
	free(d);
	free(e);
	free(f);
	free(g);
	free(body);
}


void Init() {

	int i, j, k;
	float cx, cy, cz;  // Discrete point locations in x,y,z
	int index = 0;
	int part;
	
	for (i = 0; i < NX; i++) {
		for (j = 0; j < NY; j++) {
			for (k = 0; k < NZ; k++) {
				// Set conditions - first, detect away
				cx = (i+0.5)*DX;
				cy = (j+0.5)*DY;
				cz = (k+0.5)*DZ;
				// Initial temperature is 0 everywhere
				a[index] = 0.0;     
				if (cz < 0.01) {
					// This is a body
					body[index] = 1.0;
				} else {
					part = (int)(cx/DXFIN);
					
					if (( part%2 == 1) && (cy > 0.1*H) && (cy < 0.9*H) && (cz < 0.9*W)) {
						// We are in a fin
						body[index] = 1.0;
					} else {
						// Not in a fin
						body[index] = 0.0;						
					}
				}
				if (body[index] > 0) {
					total_cells++;
				}
				index++;
			}
		}
	}
	// How many cells will actually be involved in the solution?
	printf("Identified %d cells\n", total_cells);
}


void Save_Data() {
	
	FILE *pFile;
	int i,j,k;
	int index, counter;
	float cx, cy, cz;
	float px, py, pz;
	printf("Saving....");
	
	pFile = fopen("TECPLOT.dat", "w");
	if (pFile == NULL) {
		printf("Cannot open TECPLOT.dat for writing\n");
	} else {

		// Start writing the tecplot file
		fprintf(pFile, "TITLE = \"Results\"\n");
		fprintf(pFile, "VARIABLES = \"X\", \"Y\", \"Z\", \"Temp\" \n"); 
		fprintf(pFile, "ZONE N = %d, E = %d, DATAPACKING = POINT, ZONETYPE = FEBRICK\n", 8*total_cells, total_cells); // Each cell center will be used as a point here
		index = 0;
		for (i = 0; i < NX; i++) {
			for (j = 0; j < NY; j++) {
				for (k = 0; k < NZ; k++) {	
					if (body[index] > 0) {
						// There are going to be 8 points here in our FEBRICK element
						cx = (i+0.5)*DX;
						cy = (j+0.5)*DY;
						cz = (k+0.5)*DZ;
						px = cx - 0.5*DX; py = cy - 0.5*DY; pz = cz - 0.5*DZ; 
						fprintf(pFile, "%g %g %g %g\n", px, py, pz, a[index]);
						px = cx - 0.5*DX; py = cy - 0.5*DY; pz = cz + 0.5*DZ; 
						fprintf(pFile, "%g %g %g %g\n", px, py, pz, a[index]);
						px = cx - 0.5*DX; py = cy + 0.5*DY; pz = cz + 0.5*DZ; 
						fprintf(pFile, "%g %g %g %g\n", px, py, pz, a[index]);
						px = cx - 0.5*DX; py = cy + 0.5*DY; pz = cz - 0.5*DZ; 
						fprintf(pFile, "%g %g %g %g\n", px, py, pz, a[index]);
						px = cx + 0.5*DX; py = cy - 0.5*DY; pz = cz - 0.5*DZ; 
						fprintf(pFile, "%g %g %g %g\n", px, py, pz, a[index]);
						px = cx + 0.5*DX; py = cy - 0.5*DY; pz = cz + 0.5*DZ; 
						fprintf(pFile, "%g %g %g %g\n", px, py, pz, a[index]);
						px = cx + 0.5*DX; py = cy + 0.5*DY; pz = cz + 0.5*DZ; 
						fprintf(pFile, "%g %g %g %g\n", px, py, pz, a[index]);
						px = cx + 0.5*DX; py = cy + 0.5*DY; pz = cz - 0.5*DZ; 
						fprintf(pFile, "%g %g %g %g\n", px, py, pz, a[index]);

					}
					index++;
				}
			}
		}
		index = 0;
		counter = 0;
		for (i = 0; i < NX; i++) {
			for (j = 0; j < NY; j++) {
				for (k = 0; k < NZ; k++) {	
					if (body[index] > 0) {
						fprintf(pFile, "%d %d %d %d %d %d %d %d\n", 8*counter+1, 8*counter+2,8*counter+3, 8*counter+4, 8*counter+5,8*counter+6,8*counter+7,8*counter+8);
						counter++;		
					}
					index++;
				}
			}
		}
		fclose(pFile);		
	}
	printf("Done\n");
}


void Compute_OpenMP() {

	// Compute the temperature field with OpenMP
	// First mission is to determine the neighbouring points
	int x_cell, y_cell, z_cell;
	int i, k;

	omp_set_num_threads(OMP_THREADS); // Set the number of threads
	#pragma omp parallel shared(a,b,c,d,e,f,g,body) private(i,k)
	{

	for (k = 0; k < NO_STEPS; k++) {
	
		#pragma omp for
		for (i = 0; i < N; i++) {

			x_cell = (int)(i/(NY*NZ));
			y_cell = (int)(i - x_cell*NY*NZ)/NZ;
			z_cell = i - x_cell*NY*NZ - y_cell*NZ;
	
			// Neighbours in the X direction (store in b, c)
			if (x_cell == (NX-1)) {
				// Boundary - these are fixed at 0K 
				b[i] = 0.0;
			} else {
				// Standard cell on the right
				b[i] = a[i+NY*NZ]; 
			}

			// Set left temperature
			if (x_cell == 0) {
				// Another boundary - fixed at 0K
				c[i] = 0.0;		
			} else {
				// Standard cell on the left
				c[i] = a[i-NY*NZ];
			}

			// Neighbours in the Y direction (store in d, e)
			if (y_cell == (NY-1)) {
				// Boundary, fixed at 0K
				d[i] =  0.0;
			} else {
				// Normral cell above
				d[i] = a[i+NZ];	
			}
			if (y_cell == 0) {
				// Boundary, fixed at 0K
				e[i] = 0.0;	
			} else {	
				// Normal cell below
				e[i] = a[i-NZ];
			}

			// Neighbours in the Z direction (store in f, g)
			if (z_cell == (NZ-1)) {
				// Boundary, fixed at 0K
				f[i] =  0.0;
			} else {
				// Normal cell behind
				f[i] = a[i+1];	
			}
			if (z_cell == 0) {
				// Boundary - this one has a constant temperature of 1
				g[i] = 1.0;	
			} else {	
				// Normal cell in front
				g[i] = a[i-1];
			}
		}

		#pragma omp for
		for (i = 0; i < N; i++) {
			a[i] = a[i] + body[i]*(PHI_X*(b[i] + c[i] - 2.0*a[i]) + PHI_Y*(d[i] + e[i] - 2.0*a[i]) + PHI_Z*(f[i] + g[i] - 2.0*a[i]));
		}
	}

	} // End omp loop
}

void Compute_OpenMP_AVX() {

	// Compute the temperature field with OpenMP and AVX intrinsics
	int x_cell, y_cell, z_cell;
	int i, k, index, tid;
	int N_AVX = (int)N/(8*OMP_THREADS);

	// Declare our AVX types
	__m256 *AVX_a, *AVX_b, *AVX_c, *AVX_d, *AVX_e, *AVX_f, *AVX_g, *AVX_body;
	__m256 temp1, temp2, temp3;
	__m256 AVX_2a;

	printf("Each core will perform operations on %g AVX packed elements\n", (float)N_AVX);

	omp_set_num_threads(OMP_THREADS); 	// Set the number of OpenMP threads

	#pragma omp parallel shared(a,b,c,d,e,f,g,body, AVX_PHI_X, AVX_PHI_Y, AVX_PHI_Z) private(i,k, tid, index, AVX_a, AVX_b, AVX_c, AVX_d, AVX_e, AVX_f, AVX_g, AVX_body, AVX_2a, temp1, temp2, temp3)
	{

	tid = omp_get_thread_num();
	index = tid*N_AVX;  				// Find the location within our array
	AVX_a = (__m256*)a + index;			// Pack our temperature data (a-g)
	AVX_b = (__m256*)b + index;
	AVX_c = (__m256*)c + index;
	AVX_d = (__m256*)d + index;
	AVX_e = (__m256*)e + index;
	AVX_f = (__m256*)f + index;
	AVX_g = (__m256*)g + index;
	AVX_body = (__m256*)body + index;	// Also require body variable

	for (k = 0; k < NO_STEPS; k++) {
	
		#pragma omp for
		for (i = 0; i < N; i++) {
			
			x_cell = (int)(i/(NY*NZ));
			y_cell = (int)(i - x_cell*NY*NZ)/NZ;
			z_cell = i - x_cell*NY*NZ - y_cell*NZ;

			// Neighbours in X direction	
			if (x_cell == (NX-1)) {
				b[i] = 0.0;
			} else {
				b[i] = a[i+NY*NZ];
			}
			if (x_cell == 0) {
				c[i] = 0.0;		
			} else {
				c[i] = a[i-NY*NZ];
			}

			// Neighbours in Y direction
			if (y_cell == (NY-1)) {
				d[i] =  0.0;
			} else {
				d[i] = a[i+NZ];	
			}
			if (y_cell == 0) {
				e[i] = 0.0;	
			} else {	
				e[i] = a[i-NZ];
			}

			// Neighbours in Z direction
			if (z_cell == (NZ-1)) {
				f[i] =  0.0;
			} else {
				f[i] = a[i+1];	
			}
			if (z_cell == 0) {
				g[i] = 1.0;	
			} else {	
				g[i] = a[i-1];
			}
		}

		// This part will be manually vectorized with intrinsics over N_AVX types (on each core)
		for (i = 0; i < N_AVX; i++) {
			AVX_2a = _mm256_mul_ps(AVX_a[i], AVX_TWO);  // 2a[i]
			temp1 = _mm256_add_ps(AVX_b[i], AVX_c[i]);  // b[i] + c[i]
			temp1 = _mm256_sub_ps(temp1, AVX_2a);       // b[i] + c[i] - 2a[i]
			temp1 = _mm256_mul_ps(temp1, AVX_PHI_X);    // PHI_X*(b+c-2a)

			temp2 = _mm256_add_ps(AVX_d[i], AVX_e[i]);  // d[i] + e[i]
			temp2 = _mm256_sub_ps(temp2, AVX_2a);       // d[i] + e[i] - 2a[i]
			temp2 = _mm256_mul_ps(temp2, AVX_PHI_Y);    // PHI_Y*(d+e-2a)

			temp3 = _mm256_add_ps(AVX_f[i], AVX_g[i]);  // f[i] + g[i]
			temp3 = _mm256_sub_ps(temp3, AVX_2a);       // f[i] + g[i] - 2a[i]
			temp3 = _mm256_mul_ps(temp3, AVX_PHI_Z);    // PHI_Z*(f+g-2a)

			temp1 = _mm256_add_ps(temp1, temp2);
			temp1 = _mm256_add_ps(temp1, temp3);
			temp1 = _mm256_mul_ps(temp1, AVX_body[i]);  // Final change in temperature 
			AVX_a[i] = _mm256_add_ps(AVX_a[i], temp1);
		}
		#pragma omp barrier
	}

	} // End omp loop
}



