#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "helpers.h"

#define TAG_UP   0
#define TAG_DOWN 1
#define TAG_DIFF 2
       
float* jacobi_opti(int n, int maxIter, int argc, char *argv[]) {

    int numProcs, myID, height;
    float localDiff = 0.0, globalDiff = 0.0;

    MPI_Status status;
    MPI_Request rqSendUp, rqSendDown;

    int ierrUp, ierrDown;

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

        /* Envoi des bords aux voisins */
        if(myID > 0) {
            ierrUp = MPI_Isend(new[1], n + 2, MPI_FLOAT, myID - 1, TAG_UP, MPI_COMM_WORLD, &rqSendUp);
            handle_mpi_error(ierrUp);
        }
        if(myID < numProcs - 1) {
            ierrDown = MPI_Isend(new[height], n + 2, MPI_FLOAT, myID + 1, TAG_DOWN, MPI_COMM_WORLD, &rqSendDown);
            handle_mpi_error(ierrDown);
        }

        for(int i = 2; i <= height - 1; i++) {
            for(int j = 1; j <= n; j++) {
                grid[i][j] = (new[i - 1][j] + new[i + 1][j] + new[i][j - 1] + new[i][j + 1]) * 0.25;
            }
        }

        /* Attente des envois asynchrones */
        if(myID > 0)
            MPI_Wait(&rqSendUp, &status);
        if(myID < numProcs - 1)
            MPI_Wait(&rqSendDown, &status);

        /* Réception des bords des voisins */
        if(myID < numProcs - 1) {
            memcpy(new[height + 1], grid[height + 1], sizeof(float) * (n + 2));
            ierrUp = MPI_Recv(new[height + 1], n + 2, MPI_FLOAT, myID + 1, TAG_UP, MPI_COMM_WORLD, &status);
            handle_mpi_error(ierrUp);
        }
        if(myID > 0) {
            memcpy(new[0], grid[0], sizeof(float) * (n + 2));
            ierrDown = MPI_Recv(new[0], n + 2, MPI_FLOAT, myID - 1, TAG_DOWN, MPI_COMM_WORLD, &status);
            handle_mpi_error(ierrDown);
        }
        /* Calcul des valeurs pour les bords */
        for(int j = 1; j <= n; j++) {
            grid[1][j] = (new[0][j] + new[2][j] + new[1][j - 1] + new[1][j + 1]) * 0.25;
            grid[height][j] = (new[height - 1][j] + new[height + 1][j] + new[height][j - 1] + new[height][j + 1]) * 0.25;
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
        printf("New :\n");
        print_buffer(globalNew, (n + 2) * n, n + 2);
        printf("Grid :\n");
        print_buffer(globalGrid, (n + 2) * n, n + 2);
        printf("Différence maximale = %f\n", globalDiff);
    }

    MPI_Finalize();

    return globalGrid;
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int maxIter = atoi(argv[2]);

    if(argc != 3 || n < 3 || maxIter < 1) {
        printf("Utilisation : jacobi <n> <max_iter>\nn : largeur de la grille, supérieure à 3\n max_iter : nombre d'itérations, supérieur à 1");
        exit(EXIT_FAILURE);
    }

    jacobi_opti(n, maxIter, argc, argv);
    
    return 0;
}
 
