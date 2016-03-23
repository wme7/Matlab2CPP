
#include <stdio.h>
#include <mpi.h>
#include "heat1d.h"

// Our local function declaration
void Copy_To_Friends(int rank, float **h_a);	// Distribute data to slaves
void Copy_From_Friends(int rank, float **h_a);	// Collect data from slaves
void Pass_BC_Right(int rank, float **h_a);	// Pass right-end B/C 
void Pass_BC_Left(int rank, float **h_a);	// Pass left-end B/C

int main() {
	
  MPI_Init(NULL, NULL);				// Start MPI ASAP
  int step;
  float *h_a; 					// Local array holding old temp.
  float *h_b;					// Local array, new temperature.
  float *d_a, *d_b;				// Local array for device variables
  int world_size;				// The total number of processes
  int world_rank;				// Rank of each process (i.e. ID)
  char processor_name[MPI_MAX_PROCESSOR_NAME];	// Name of each processor
  int name_len;					// Length of the name
	
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);	// Get the number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 	// Get the rank of the process
  MPI_Get_processor_name(processor_name, &name_len); // Get the name

  // Print a message saying hello from each processor and rank:
  printf("Commence Simulation: processor %s, rank %d out of %d processors\n",
	 processor_name, world_rank, world_size);

  // Allocate memory
  Allocate_Memory(world_rank, &h_a, &d_a, &d_b);

  // Rank 0 will initialize proceedings 
  if (world_rank == 0) Init(h_a);

  // Barrier here, we want to make sure all is ready
  MPI_Barrier(MPI_COMM_WORLD);

  Copy_To_Friends(world_rank, &h_a);		// Send data to slaves.
  MPI_Barrier(MPI_COMM_WORLD);			// Make sure we are ready 

  // =============== BOUNDARY MANAGEMENT =================
  // Copy end points of regions between slaves. Right First, then left
  if ((DEBUG) && (world_rank == 0)) printf("==Passing Right==\n"); 
  Pass_BC_Right(world_rank, &h_a);
  MPI_Barrier(MPI_COMM_WORLD);
  if ((DEBUG) && (world_rank == 0)) printf("==Passing Left==\n");
  Pass_BC_Left(world_rank, &h_a);
  // The outer-most ends of our 1D problem have global boundaries.
  // These are controlled by the threads holding these bounds.
  if (world_rank == 1) h_a[  0 ] = h_a[ 1];  	// Neumann Condition
  if (world_rank == 4) h_a[NP+1] = h_a[NP];	// Neumann Conditions
  // =============== END BOUNDARY MANAGEMENT =================
  
  // Copy h_a to the GPU in preperation for computation
  if (USE_GPU) Copy_All_To_GPU(world_rank, &h_a, &d_a, &d_b);
  
  // Transient time loop
  for (step = 0; step < 150; step++) {
    if (USE_GPU) {
      // Call the wrapping function for our GPU kernel
      GPU_Compute(world_rank, &d_a, &d_b);  
      MPI_Barrier(MPI_COMM_WORLD);	// Make sure each thread is ready
    } else {
      // Using CPU only
      CPU_Compute(world_rank, h_a, h_b);
      MPI_Barrier(MPI_COMM_WORLD);	// Make sure each thread is ready
    }
    // =============== BOUNDARY MANAGEMENT =================
    if (USE_GPU) GPU_Send_Ends(world_rank, &h_a, &d_a);
    if ((DEBUG) && (world_rank == 0)) printf("==Passing Right==\n"); 
    Pass_BC_Right(world_rank, &h_a);
    MPI_Barrier(MPI_COMM_WORLD);
    if ((DEBUG) && (world_rank == 0)) printf("==Passing Left==\n");
    Pass_BC_Left(world_rank, &h_a);
    // Take care of our global boundary conditions (update values)
    if (world_rank == 1) h_a[0] = h_a[1];  		// Neumann Condition
    if (world_rank == 4) h_a[NP+1] = h_a[NP];	// Neumann Conditions
    if (USE_GPU) GPU_Recieve_Ends(world_rank, &h_a, &d_a);
    // =============== END BOUNDARY MANAGEMENT =================
    MPI_Barrier(MPI_COMM_WORLD); // Make sure we are all ready
  }
  
  // Copy entire data set from GPU to host variables
  if (USE_GPU) Copy_All_From_GPU(world_rank, &h_a, &d_a, &d_b);  // Copies d_b into h_a on each rank
  
  // Copy data from the slaves to the main thread.
  Copy_From_Friends(world_rank, &h_a);	// Sends h_a from each rank into the whole h_a on rank 0.
  
  // Save our results to file
  if (world_rank == 0) Save_Result(h_a);
  
  // Free the memory
  Free_Memory(world_rank, &h_a, &d_a, &d_b);
  
  // We must conclude our parallel work before continuing.
  MPI_Finalize();
  
  return 0;
}


void Copy_To_Friends(int rank, float **h_a) {
  // Distribute information from the main thread to the slaves
  if (rank == 0) {
    // Send the first NP = (N/4) elements from h_a  (elements 0,1,2..24)
    MPI_Send(*h_a, NP, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    // Send the next NP elements from h_a (elements 25,26...49) 
    MPI_Send(*h_a+NP, NP, MPI_FLOAT, 2, 0, MPI_COMM_WORLD);
    // Send the next NP elements from h_a (elements 50, 51, ...74) 
    MPI_Send(*h_a+2*NP, NP, MPI_FLOAT, 3, 0, MPI_COMM_WORLD);
    // Send the next NP elements from h_a (elements 75, 76, ...99) 
    MPI_Send(*h_a+3*NP, NP, MPI_FLOAT, 4, 0, MPI_COMM_WORLD);
  } else {
    // This is a slave thread - prepare to recieve data.
    // This data starts in the 2nd element (element 1) since 0 contains BC data.
    MPI_Recv(*h_a+1, NP, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void Copy_From_Friends(int rank, float **h_a) {
  // Collect data from slaves if rank = 0. Otherwise, send it.
  // Let's go - Rank 1 --> Main Thread (0) First
  if (rank == 1) {
    MPI_Send(*h_a+1, NP, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  } else if (rank == 0) {
    MPI_Recv(*h_a, NP, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  // Once rank 0 is ready, we will continue.
  MPI_Barrier(MPI_COMM_WORLD);
  
  // 2 --> 0 Now
  if (rank == 2) {
    MPI_Send(*h_a+1, NP, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  } else if (rank == 0) {
    MPI_Recv(*h_a+NP, NP, MPI_FLOAT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  // 3 --> 0 Now
  if (rank == 3) {
    MPI_Send(*h_a+1, NP, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  } else if (rank == 0) {
    MPI_Recv(*h_a+2*NP, NP, MPI_FLOAT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  // 4 --> 0 Now
  if (rank == 4) {
    MPI_Send(*h_a+1, NP, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  } else if (rank == 0) {
    MPI_Recv(*h_a+3*NP, NP, MPI_FLOAT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  MPI_Barrier(MPI_COMM_WORLD); // Add one final barrier for good measure.
}


void Pass_BC_Right(int rank, float **h_a) {
  // Each rank (except rank 0) passes the right end of its data (element NP) to the 1st element
  // of its right hand side neighbour region
  // We have to stagger this process - you won't pass and recieve at the same time in this code!
  //   1	|   2	|   3	|   4	|   First, pass 1 to 2 and 3 to 4.
  //   ----->    	|    ----->	|
  //	|	|	|	|
  if ((rank == 1) || (rank == 3)) {
    // This rank is sending one value
    MPI_Send(*h_a+NP, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
  } 
  if ((rank == 2) || (rank == 4)) {
    // This rank is recieving one value
    MPI_Recv(*h_a, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  
  MPI_Barrier(MPI_COMM_WORLD); // Make sure we are all ready.
  
  //   1	|   2	|   3	|   4	|   Now send 2 to 3.
  //   	|    ------>   	|	|
  //	|	|	|	|
  
  if (rank == 2) {
    // Sending one value
    MPI_Send(*h_a+NP, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
  } else if (rank == 3) {
    // Recieving one value
    MPI_Recv(*h_a, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
		
}

void Pass_BC_Left(int rank, float **h_a) {
  
  // Each rank (except rank 0) passes the left end of its data 
  // to the LAST element of its left hand side neighbour region
  //   1	|   2	|   3	|    4	|   First, pass 1 to 2 and 3 to 4.
  //    <-----    |     <-----	|
  //		|		|		|		|
  if ((rank == 2) || (rank == 4)) {
    // These ranks are passing 1 value
    MPI_Send(*h_a+1, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
  } else if ((rank == 1) || (rank == 3)) {
    // These ranks are recieving
    MPI_Recv(*h_a+NP+1, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);	// Make sure we are ready

  //    |   1	|   2	|   3	|    4	|   Now send 2 to 3. 
  //   	|     <-----   	|		|
  //	|		|		|		|
  
  if (rank == 3) {
    MPI_Send(*h_a+1, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
  } else if (rank == 2) {
    MPI_Recv(*h_a+NP+1, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}
