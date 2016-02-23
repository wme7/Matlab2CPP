/* Tutorial #2: Vector Addition using AVX intrinsic functions
   Written by: Matthew Smith, Department of Mechanical Engineering, NCKU
   For corrections or queries: msmith@mail.ncku.edu.tw

   Function Description
   
   Allocate_Memory():   	Allocate the memory for variables a,b and c.
   Free_Memory():			Free the variables a,b and c from memory.
   Init();					Set the values of a and b.
   Compute_Ordinary();		Compute the sum c = a + b using standard code.
   Compute_AVX();		    Compute the sum c = a + b using AVX intrinsics

   Last Updated: 1st Oct, 2015

   This code is subject to the disclaimers found on hshclncku.com 
   Use at your own risk.
*/	

#include <stdio.h>
#include <immintrin.h>
#include <math.h>

#define N 32000     // An integer N describing the problem size

// Create some variables a,b,c
float *a, *b, *c;

// Declare our functions
void Allocate_Memory();
void Free_Memory();
void Init();
void Compute_AVX();
void Compute_Ordinary();

int main() {
	int i;
	Allocate_Memory();
	Init();
	Compute_AVX();
	for (i = 0; i < 10; i++) {
		printf("c[%d] = %g\n", i, c[i]);
	}
	Compute_Ordinary();
	for (i = 0; i < 10; i++) {
		printf("c[%d] = %g\n", i, c[i]);
	}
	Free_Memory();

}

void Allocate_Memory() {
	size_t alignment = 32;
	int error;
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

void Compute_AVX() {
	int i;
	int N_AVX = N/8;    // 8 floats packed into each AVX type
	__m256 *AVX_a, *AVX_b, *AVX_c;
	AVX_a = (__m256*)a; AVX_b = (__m256*)b; AVX_c = (__m256*)c;
	for (i = 0; i < N_AVX; i++) {
		AVX_c[i] = _mm256_add_ps(AVX_a[i], AVX_b[i]);
	}
}


