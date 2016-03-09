// Assume we will create 5 threads using OpenMPI
// Thread 0 will do nothing but coordinate
// Threads 1-4 will have 25+2 elements each.

#define N 100		// Total problem size
#define NP 25		// 100/4 (shared across 4 ranks)
#define DEBUG 1		// Print debug messages (set to 0 to turn off)
#define PHI 0.1		// Our CFL for the HT problem
#define USE_GPU 1	// Use GPU (set to 0 to turn off)

void Allocate_Memory(int rank, float **h_a, float **d_a, float **d_b); 
void Free_Memory(int rank, float **h_a, float **d_a, float **d_b);

void Copy_All_To_GPU(int rank, float **h_a, float **d_a, float **d_b);
void Copy_All_From_GPU(int rank, float **h_a, float **d_a, float **d_b);

void CPU_Compute(int rank, float *h_a, float *h_b);

void GPU_Compute(int rank, float **d_a, float **d_b);
void GPU_Send_Ends(int rank, float **h_a, float  **d_a);
void GPU_Recieve_Ends(int rank, float **h_a, float  **d_a);
