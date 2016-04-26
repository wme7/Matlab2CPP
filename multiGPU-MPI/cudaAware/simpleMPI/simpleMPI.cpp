

#include <mpi.h>
#include "simpleMPI.h"

/*************/
/* Host code */
/*************/

// No CUDA here, only MPI
int main(int argc, char *argv[]){

    // Dimensions of the dataset
    int blockSize = 256;
    int gridSize = 1000000;
    int dataSizePerNode = gridSize * blockSize;

    // Initialize MPI state
    MPI_Init(&argc, &argv);

    // Get our MPI node number and node count
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Generate some random numbers on the root node (node 0)
    int dataSizeTotal = dataSizePerNode * size;
    float *dataRoot = NULL;

    // Root node initializa data
    if (rank == 0) {
	printf("  Running on %d nodes\n",size);
        dataRoot = new float[dataSizeTotal];
        initData(dataRoot, dataSizeTotal);
    }

    // Allocate a buffer on each node
    float *dataNode = new float[dataSizePerNode];

    // Dispatch a portion of the input data to each node
    MPI_Scatter(dataRoot,dataSizePerNode,MPI_FLOAT,
                dataNode,dataSizePerNode,MPI_FLOAT,0,MPI_COMM_WORLD);

    // Delete data form Root
    if (rank == 0) {delete [] dataRoot;}

    // On each node, run computation on GPU
    computeGPU(dataNode, blockSize, gridSize);

    // Reduction to the root node, computing the sum of output elements
    float sumRoot, sumNode = sum(dataNode, dataSizePerNode);
    MPI_Reduce(&sumNode, &sumRoot, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        float average = sumRoot / dataSizeTotal;
	printf("  Average of square roots is: %1.2f\n",average);
    }

    // Cleanup
    delete [] dataNode;
    MPI_Finalize();

    // print exit message
    if (rank == 0) printf("  TEST PASSED\n");

    return 0;
}

