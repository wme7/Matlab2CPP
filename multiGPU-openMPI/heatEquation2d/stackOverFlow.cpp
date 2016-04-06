#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

/*
 This is a version with integers, rather than char arrays, presented in this
 very good answer: http://stackoverflow.com/a/9271753/2411320
 
 It will initialize the 2D array, scatter it, increase every value by 1 and then gather it back.

 compile as:
 $ mpic++ example.cpp 
 $ mpirun -np 4 a.out 

*/

int malloc2D(int ***array, int n, int m) {
    int i;
    /* allocate the n*m contiguous items */
    int *p = (int*)malloc(n*m*sizeof(int));
    if (!p) return -1;

    /* allocate the row pointers into the memory */
    (*array) = (int**)malloc(n*sizeof(int*));
    if (!(*array)) {
       free(p);
       return -1;
    }

    /* set up the pointers into the contiguous memory */
    for (i=0; i<n; i++)
       (*array)[i] = &(p[i*m]);

    return 0;
}

int free2D(int ***array) {
    /* free the memory - the first element of the array is at the start */
    free(&((*array)[0][0]));

    /* free the pointers into the memory */
    free(*array);

    return 0;
}

int main(int argc, char **argv) {
    int **global, **local;
    const int gridsize=8; // size of grid
    const int procgridsize=2;  // size of process grid
    int rank, size;        // rank of current process and no. of processes
    int i, j, p;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if (size != procgridsize*procgridsize) {
        fprintf(stderr,"%s: Only works with np=%d for now\n", argv[0], procgridsize);
        MPI_Abort(MPI_COMM_WORLD,1);
    }


    if (rank == 0) {
        /* fill in the array, and print it */
        malloc2D(&global, gridsize, gridsize);
        int counter = 0;
        for (i=0; i<gridsize; i++) {
            for (j=0; j<gridsize; j++)
                global[i][j] = ++counter;
        }


        printf("Global array is:\n");
        for (i=0; i<gridsize; i++) {
            for (j=0; j<gridsize; j++) {
                printf("%2d ", global[i][j]);
            }
            printf("\n");
        }
    }
    //return;

    /* create the local array which we'll process */
    malloc2D(&local, gridsize/procgridsize, gridsize/procgridsize);

    /* create a datatype to describe the subarrays of the global array */
    int sizes[2]    = {gridsize, gridsize};         /* global size */
    int subsizes[2] = {gridsize/procgridsize, gridsize/procgridsize};     /* local size */
    int starts[2]   = {0,0};                        /* where this one starts */
    MPI_Datatype type, subarrtype;
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &type);
    MPI_Type_create_resized(type, 0, gridsize/procgridsize*sizeof(int), &subarrtype);
    MPI_Type_commit(&subarrtype);

    int *globalptr=NULL;
    if (rank == 0)
        globalptr = &(global[0][0]);

    /* scatter the array to all processors */
    int sendcounts[procgridsize*procgridsize];
    int displs[procgridsize*procgridsize];

    if (rank == 0) {
        for (i=0; i<procgridsize*procgridsize; i++)
            sendcounts[i] = 1;
        int disp = 0;
        for (i=0; i<procgridsize; i++) {
            for (j=0; j<procgridsize; j++) {
                displs[i*procgridsize+j] = disp;
                disp += 1;
            }
            disp += ((gridsize/procgridsize)-1)*procgridsize;
        }
    }


    MPI_Scatterv(globalptr, sendcounts, displs, subarrtype, &(local[0][0]),
                 gridsize*gridsize/(procgridsize*procgridsize), MPI_INT,
                 0, MPI_COMM_WORLD);

    /* now all processors print their local data: */

    for (p=0; p<size; p++) {
        if (rank == p) {
            printf("Local process on rank %d is:\n", rank);
            for (i=0; i<gridsize/procgridsize; i++) {
                putchar('|');
                for (j=0; j<gridsize/procgridsize; j++) {
                    printf("%2d ", local[i][j]);
                }
                printf("|\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* now each processor has its local array, and can process it */
    for (i=0; i<gridsize/procgridsize; i++) {
        for (j=0; j<gridsize/procgridsize; j++) {
            local[i][j] += 1; // increase by one the value
        }
    }

    /* it all goes back to process 0 */
    MPI_Gatherv(&(local[0][0]), gridsize*gridsize/(procgridsize*procgridsize),  MPI_INT,
                 globalptr, sendcounts, displs, subarrtype,
                 0, MPI_COMM_WORLD);

    /* don't need the local data anymore */
    free2D(&local);

    /* or the MPI data type */
    MPI_Type_free(&subarrtype);

    if (rank == 0) {
        printf("Processed grid:\n");
        for (i=0; i<gridsize; i++) {
            for (j=0; j<gridsize; j++) {
                printf("%2d ", global[i][j]);
            }
            printf("\n");
        }

        free2D(&global);
    }


    MPI_Finalize();

    return 0;
}
