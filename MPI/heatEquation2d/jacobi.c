#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "helpers.h"

#define TAG_UP   0
#define TAG_DOWN 1
#define TAG_DIFF 2
 
result jacobi(int n, int maxIter, int myID, int numProcs) {

    result myResult;

    int height;
    float localDiff = 0.0, globalDiff = 0.0;

    MPI_Status status;
    MPI_Request rqSendUp, rqSendDown, rqRecvUp, rqRecvDown;
    
    int ierrUp, ierrDown;
       
    /* Calcul de la hauteur de la tranche */
    height = n / numProcs;

    /* Allocation des deux tableaux pour la tranche */
    float **grid = (float**) malloc((height + 2) * sizeof(float*));
    float **new = (float**) malloc((height + 2) * sizeof(float*));

    float *globalGrid;
    float *globalNew;

    for(int j = 0; j < height + 2; j++) {
        grid[j] = (float*) malloc((n + 2) * sizeof(float));
        new[j] = (float*) malloc((n + 2) * sizeof(float));
    }
    
    /* Allocation des tableaux de résultat */
    if(myID == 0) {
        globalGrid = (float*) malloc((n + 2) * height * numProcs * sizeof(float));
        globalNew = (float*) malloc((n + 2) * height * numProcs * sizeof(float));
    }

    initialize(grid, n, height, numProcs, myID);
    initialize(new, n, height, numProcs, myID);

    /* Synchronisation globale */
    MPI_Barrier(MPI_COMM_WORLD);

    for(int t = 0; t < maxIter; t = t + 2) {
        for(int i = 1; i <= height; i++) {
            for(int j = 1; j <= n; j++) {
                new[i][j] = (grid[i - 1][j] + grid[i + 1][j] + grid[i][j - 1] + grid[i][j + 1]) * 0.25;
            }
        }

        /* Envoi/réception des bords */
        if(myID < numProcs - 1) {
            ierrDown = MPI_Isend(new[height], n + 2, MPI_FLOAT, myID + 1, TAG_DOWN, MPI_COMM_WORLD, &rqSendDown);
            handle_mpi_error(ierrDown);
            ierrUp = MPI_Irecv(new[height+1], n + 2, MPI_FLOAT, myID + 1, TAG_UP, MPI_COMM_WORLD, &rqRecvUp);
            handle_mpi_error(ierrUp);
        }
        if(myID > 0) {
            ierrUp = MPI_Isend(new[1], n + 2, MPI_FLOAT, myID - 1, TAG_UP, MPI_COMM_WORLD, &rqSendUp);
            handle_mpi_error(ierrUp);
            ierrDown = MPI_Irecv(new[0], n + 2, MPI_FLOAT, myID - 1, TAG_DOWN, MPI_COMM_WORLD, &rqRecvDown);
            handle_mpi_error(ierrDown);
        }

        /* Attente des envois asynchrones */
        if(myID < numProcs - 1) {
            MPI_Wait(&rqSendDown, &status);
            MPI_Wait(&rqRecvUp, &status);
        }
        if(myID > 0) {
            MPI_Wait(&rqRecvDown, &status);
            MPI_Wait(&rqSendUp, &status);
        }

        for(int i = 1; i <= height; i++) {
            for(int j = 1; j <= n; j++) {
                grid[i][j] = (new[i - 1][j] + new[i + 1][j] + new[i][j - 1] + new[i][j + 1]) * 0.25;
            }
        }
        
        /* Envoi/réception des bords */
        if(myID < numProcs - 1) {
            ierrDown = MPI_Isend(grid[height], n + 2, MPI_FLOAT, myID + 1, TAG_DOWN, MPI_COMM_WORLD, &rqSendDown);
            handle_mpi_error(ierrDown);
            ierrUp = MPI_Irecv(grid[height+1], n + 2, MPI_FLOAT, myID + 1, TAG_UP, MPI_COMM_WORLD, &rqRecvUp);
            handle_mpi_error(ierrUp);
        }
        if(myID > 0) {
            ierrUp = MPI_Isend(grid[1], n + 2, MPI_FLOAT, myID - 1, TAG_UP, MPI_COMM_WORLD, &rqSendUp);
            handle_mpi_error(ierrUp);
            ierrDown = MPI_Irecv(grid[0], n + 2, MPI_FLOAT, myID - 1, TAG_DOWN, MPI_COMM_WORLD, &rqRecvDown);
            handle_mpi_error(ierrDown);
        }

        /* Attente des envois asynchrones */
        if(myID < numProcs - 1) {
            MPI_Wait(&rqSendDown, &status);
            MPI_Wait(&rqRecvUp, &status);
        }
        if(myID > 0) {
            MPI_Wait(&rqRecvDown, &status);
            MPI_Wait(&rqSendUp, &status);
        }
    }

    /* Récupération des résultats */ 
    MPI_Gather(grid[1], (n + 2) * height, MPI_FLOAT, globalGrid, (n + 2) * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(new[1], (n + 2) * height, MPI_FLOAT, globalNew, (n + 2) * height, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Calcul du maximum local */
    for(int i = 1; i <= height; i++) {
        for(int j = 1; j <= n; j++) {
            localDiff = fmax(localDiff, fabs(grid[i][j] - new[i][j]));
        }
    }
    
    /* Calcul du maximum global */
    MPI_Reduce(&localDiff, &globalDiff, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        
    if(myID == 0) {
        myResult.grid = globalGrid;
        myResult.diff = globalDiff;    
    }

    return myResult;
}

int main(int argc, char *argv[]) {

    int myID, numProcs;
    int n = atoi(argv[1]);
    int maxIter = atoi(argv[2]);
    result myResult;

    /* Initialisation de MPI */
    MPI_Init( &argc, &argv );

    /* Détermination de l'id du process courant */
    MPI_Comm_rank( MPI_COMM_WORLD, &myID );

    /* Détermination du nombre de process */
    MPI_Comm_size( MPI_COMM_WORLD, &numProcs );

    if(n % numProcs != 0) {
        printf("n doit être un multiple de %d", numProcs);
        exit(EXIT_FAILURE);
    }
    
    if(argc != 3 || n < 3 || maxIter < 1) {
        printf("Utilisation : jacobi <n> <max_iter>\nn : largeur de la grille, supérieure à 3\nmax_iter : nombre d'itérations, supérieur à 1\n");
        exit(EXIT_FAILURE);
    }
    
    myResult = jacobi(n, maxIter, myID, numProcs);
    
    if(myID == 0) {
        printf("Grid :\n");
        print_buffer(myResult.grid, (n + 2) * n, n + 2);
        printf("Différence maximale = %f\n", myResult.diff);
    }    

    MPI_Finalize(); 
    
    return 0;
}

