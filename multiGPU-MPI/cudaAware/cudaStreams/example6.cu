
#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 128
#define gpuErrchk(error) __checkCuda(error, __FILE__, __LINE__)
#define iDivUp(x,y) (((x)+(y)-1)/(y)) // define No. of blocks/warps

/*********************************************/
/* A method for checking error in CUDA calls */
/*********************************************/
inline void __checkCuda(cudaError_t error, const char *file, const int line) {
    //#if defined(DEBUG) || defined(_DEBUG)
    if (error != cudaSuccess) {
        printf("checkCuda error at %s:%i: %s\n", file, line, cudaGetErrorString(cudaGetLastError()));
        exit(-1);
    }
    //#endif
    return;
}

/*******************/
/* KERNEL FUNCTION */
/*******************/
__global__ void kernelFunction(double * __restrict__ d_data, const unsigned int NperGPU) {

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < NperGPU) {
        for (int k = 0; k < 1000; k++) d_data[tid] = d_data[tid] * d_data[tid];
    }

}

/******************/
/* PLAN STRUCTURE */
/******************/
typedef struct {
    int width;
    int height;
    double *d_elements;
    double *h_elements;
    //cudaStream_t stream;
} Matrix;

/*****************/
/* DATA CREATION */
/*****************/

// void matrix_init(device *plan, int thisSize) {
//   // initialize size and capacity
//   plan->size = thisSize;

//   // allocate memory for plan->d_data
//   plan->d_data = (double *)malloc(plan->size*sizeof(double));

//   // initialize first 10 elements
//   for (int i = 0; i < 10; i++) plan->d_data[i] = i; 
// }

// void data_print(device *plan, int thisSize) {
//   // print first 10 elements
//   printf("data : \n");
//   for (int i = 0; i < 10; i++) printf("%1.1f ",plan->d_data[i]); 
//   printf("\n");
// }

// void data_free(device *plan) {
//   free(plan->d_data);
// }

/********/
/* MAIN */
/********/
int main() {

    const int numGPUs   = 2;
    const int rows      = 500;
    const int cols      = 1000;
    const int NperGPU   = rows*cols;
    //const int N         = NperGPU*numGPUs;
    Matrix plan[numGPUs];

    //device plan[2]; 
    // for (int i = 0; i < 2; ++i)
    // {
    //     data_init(&plan[i],NperGPU);
    //     data_print(&plan[i],NperGPU);
    //     data_free(&plan[i]);
    // }

    // initialize arrays in gpus
    for (int k = 0; k < numGPUs; k++)
    {
        gpuErrchk(cudaSetDevice(k));
        gpuErrchk(cudaMalloc(&(plan[k].d_elements), NperGPU*sizeof(double)));
        gpuErrchk(cudaMallocHost((void **)&(plan[k].h_elements), NperGPU*sizeof(double)));
        //gpuErrchk(cudaStreamCreate(&(plan[k].stream)));
    }

    // initialize input matrix in host
    //double *inputMatrix = (double *)malloc(N*sizeof(double));

    // --- "Depth-first" approach - but using default stream
    for (int k = 0; k < numGPUs; k++) 
    {
        gpuErrchk(cudaSetDevice(k));
        gpuErrchk(cudaMemcpyAsync(plan[k].d_elements, plan[k].h_elements, NperGPU*sizeof(double), cudaMemcpyHostToDevice));
        kernelFunction<<<iDivUp(NperGPU, BLOCKSIZE), BLOCKSIZE>>>(plan[k].d_elements, NperGPU);
        gpuErrchk(cudaMemcpyAsync(plan[k].h_elements, plan[k].d_elements, NperGPU*sizeof(double), cudaMemcpyDeviceToHost));
    }

    gpuErrchk(cudaDeviceReset());
}
