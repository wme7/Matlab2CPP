
#include "Jacobi.h"

/**
 * @file Jacobi.c
 * @brief This contains the application entry point
 */

/**
 * @brief The application entry point
 *
 * @param[in] argc	The number of command-line arguments
 * @param[in] argv	The list of command-line arguments
 */
int main(int argc, char ** argv)
{
	MPI_Comm cartComm = MPI_COMM_WORLD;
	int rank, size;
	int neighbors[4];
	int2 domSize, topSize, topIndex;	
	int useFastSwap;
	
	double timerStart = 0.0;			
	double avgTransferTime = 0.0;
	int iterations = 0;
	
	real * devBlocks[2] 	= {NULL, NULL};		
	real * devSideEdges[2] 	= {NULL, NULL}; 	
	real * devHaloLines[2] 	= {NULL, NULL};
	real * hostSendLines[2] = {NULL, NULL};
	real * hostRecvLines[2] = {NULL, NULL};
	real * devResidue 	= NULL;

	cudaStream_t copyStream = NULL;
	
	// Initialize the MPI process
	Initialize(&argc, &argv, &rank, &size);

	// Extract topology and domain dimensions from the command-line arguments
	if (ParseCommandLineArguments(argc, argv, rank, size, &domSize, &topSize, &useFastSwap) == STATUS_OK)
	{	
		// Map the MPI ranks to corresponding GPUs and the desired topology
		if (ApplyTopology(&rank, size, &topSize, neighbors, &topIndex, &cartComm) == STATUS_OK)
		{
			// Initialize the data for the current MPI process
			InitializeDataChunk(topSize.y, topIndex.y, &domSize, neighbors, &copyStream, devBlocks, devSideEdges, 
				devHaloLines, hostSendLines, hostRecvLines, &devResidue);
			
			// Run the Jacobi computation
			PreRunJacobi(cartComm, rank, size, &timerStart);
			RunJacobi(cartComm, rank, size, &domSize, &topIndex, neighbors, useFastSwap, devBlocks, devSideEdges, 
				devHaloLines, hostSendLines, hostRecvLines, devResidue, copyStream, &iterations, &avgTransferTime);
			PostRunJacobi(cartComm, rank, size, &topSize, &domSize, iterations, useFastSwap, timerStart, avgTransferTime);
		}
	}

	// Finalize the MPI process
	Finalize(devBlocks, devSideEdges, devHaloLines, hostSendLines, hostRecvLines, devResidue, copyStream);

	return STATUS_OK;
}
