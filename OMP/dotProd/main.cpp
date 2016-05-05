
#include <stdio.h>
#include <omp.h>

#define N 160

int main() {

	// Declare our variables first
	float A[N], B[N], C[N];
	float sum;
	int i, tid;

    // Initialize A and B
	for (i = 0; i < N; i++) {
	  A[i] = 1; 
	  B[i] = (float)i;
	}
	
	// Serial vector multplication (C=A*B)
	for (i = 0; i < N; i++) {
	  C[i] = A[i]*B[i];
	}
	// Serial Reduction (sum of C)
	sum = 0.0;
	for (i = 0; i < N; i++) {
	  sum = sum + C[i];
	}
	printf("Serial result = %g\n", sum);

	// Parallel using OpenMP
	sum = 0.0;
	omp_set_num_threads(32);
	#pragma omp parallel shared(A,B,C,sum) private(i,tid)
	{
		tid = omp_get_thread_num();
		printf("Hello from thread %d\n", tid);

		// Parallel vector multplication
		#pragma omp for 
		for (i = 0; i < N; i++) {
			C[i] = A[i]*B[i];
		}
		// Parallel reduction for the sum
		#pragma omp for reduction (+:sum)
		for (i = 0; i < N; i++) {
			sum = sum + C[i];
		}
	}
	printf("Parallel result = %g\n", sum);
	return 0;
}
