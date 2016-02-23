/* Tutorial #1: Vector Addition using SSE intrinsic functions
   Written by: Matthew Smith, Department of Mechanical Engineering, NCKU
   For corrections or queries: msmith@mail.ncku.edu.tw

   Function Description
   
   Allocate_Memory():   	Allocate the memory for variables a,b and c.
   Free_Memory():			Free the variables a,b and c from memory.
   Init();					Set the values of a and b.
   Compute_Ordinary();		Compute the sum c = a + b using standard code.
   Compute_SSE();		    Compute the sum c = a + b using SSE intrinsics

   Last Updated: 1st Oct, 2015

   This code is subject to the disclaimers found on hshclncku.com 
   Use at your own risk.
*/	

#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <xmmintrin.h>

#define N 32000					// The number of elements in our arrays a,b and c

float *a, *b, *c;				// Global pointers to arrays

void Allocate_Memory();			// Function descriptions are shown above
void Free_Memory();
void Init();
void Compute_Ordinary();
void Compute_SSE();

int main() {
	int i;
	Allocate_Memory();
	Init();
	Compute_SSE();
	for (i = 0; i < 10; i++) {
		printf("c[%d] = %g\n", i, c[i]); // Check the SSE result for 10 elements
	}
	Compute_Ordinary();
	for (i = 0; i < 10; i++) {
		printf("c[%d] = %g\n", i, c[i]); // Check the standard code for 10 elements
	}
	Free_Memory();
	return 0;
}

void Allocate_Memory() {

	size_t alignment = 32;
	int error; // Error code - will be 0 if allocation is OK
	error = posix_memalign((void**)&a, alignment, N*sizeof(float)); 
	error = posix_memalign((void**)&b, alignment, N*sizeof(float));
	error = posix_memalign((void**)&c, alignment, N*sizeof(float));
}

void Free_Memory() {
	free(a);
	free(b);
	free(c);
}

void Init() {
	int i;
	for (i = 0; i < N; i++) {
		a[i] = (float)i;
		b[i] = sqrtf(i);
	}
}

void Compute_Ordinary() {
	int i;
	for (i = 0; i < N; i++) {
		c[i] = a[i] + b[i];
	}

}

void Compute_SSE() {
	int i;
	int N_SSE = N/4;
	__m128 *SSE_a, *SSE_b, *SSE_c;
	SSE_a = (__m128*)a;   SSE_b = (__m128*)b;   SSE_c = (__m128*)c;
	for (i = 0; i < N_SSE; i++) {
		SSE_c[i] = _mm_add_ps(SSE_a[i], SSE_b[i]);
	}
}

