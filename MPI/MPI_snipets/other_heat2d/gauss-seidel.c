#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "helpers.h"

#define TAG_UP   0
#define TAG_DOWN 1
#define TAG_DIFF 2

result gauss_seidel(int n, int maxIter, int myID, int numProcs) {

    result myResult;
 
    int height;
    float diff = 0.0, previousDiff = 0.0;
    float globalDiff = 0.0;
    
    MPI_Status status;
    MPI_Request rqSendUp, rqSendDown, rqRecvUp, rqRecvDown;

    /* Calcul de la hauteur de la tranche */
    height = n / numProcs;

    /* Allocation du tableau pour la tranche */
    float **grid = (float**) malloc((height + 2) * sizeof(float));
    float *globalGrid;

    for(int j = 0; j < height + 2; j++) {
        grid[j] = (float*) malloc((n + 2) * sizeof(float));
    }

    if(myID == 0) {
 	    globalGrid = (float*) malloc((n + 2) * height * numProcs * sizeof(float));
    }

    initialize(grid, n, height, numProcs, myID);
    int jStart;

    /* Synchronisation globale */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Calcul des nouvelles valeurs pour les points rouges */
    for(int t = 0; t < maxIter; t++) {
        /* Initialisation à 0.0 au début de chaque itération pour recommencer le calcul de la différence maximale */
        diff = 0.0;
        for(int i = 1; i <= height; i++) {
            /* Ligne impaire */
            if(i % 2 == 1) jStart = 1; 
            /* Ligne paire */
            else jStart = 2;
            
            for(int j = jStart; j <= n; j+=2) {
                previousDiff = grid[i][j];
                grid[i][j] = (grid[i - 1][j] + grid[i + 1][j] + grid[i][j - 1] + grid[i][j + 1]) * 0.25;
                diff = fmax(diff, fabs(grid[i][j] - previousDiff));
            }
        }

        /* Envoi des bords des points rouges aux voisins */
        if(myID > 0) {
	        MPI_Isend(grid[1], n+2, MPI_FLOAT, myID - 1, TAG_UP, MPI_COMM_WORLD, &rqSendUp);
            MPI_Irecv(grid[0], n+2, MPI_FLOAT, myID - 1, TAG_DOWN, MPI_COMM_WORLD, &rqRecvDown);
	    }
        if(myID < numProcs - 1) {
            MPI_Isend(grid[height], n+2, MPI_FLOAT, myID + 1, TAG_DOWN, MPI_COMM_WORLD, &rqSendDown);
            MPI_Irecv(grid[height+1], n+2, MPI_FLOAT, myID + 1, TAG_UP, MPI_COMM_WORLD, &rqRecvUp);
	    }

        /* Attente des envois asynchrones */
        if(myID > 0) {
            MPI_Wait(&rqSendUp, &status);
            MPI_Wait(&rqRecvDown, &status);
        }
        if(myID < numProcs - 1) {
            MPI_Wait(&rqSendDown, &status);
            MPI_Wait(&rqRecvUp, &status);
        }
        
        /* Calcul des nouvelles valeurs pour les points noirs */
        for(int i = 1; i <= height; i++) {
            /* Ligne impaire */
            if(i % 2 == 1) jStart = 2;
            /* Ligne paire */
            else jStart = 1;
            for(int j = jStart; j <= n; j+=2) {
                previousDiff = grid[i][j];
                grid[i][j] = (grid[i - 1][j] + grid[i + 1][j] + grid[i][j - 1] + grid[i][j + 1]) * 0.25;
                diff = fmax(diff, fabs(grid[i][j] - previousDiff));
            }
        
        }

        /* Envoi des bords des points noirs aux voisins */
        if(myID > 0) {
	        MPI_Isend(grid[1], n+2, MPI_FLOAT, myID - 1, TAG_UP, MPI_COMM_WORLD, &rqSendUp);
            MPI_Irecv(grid[0], n+2, MPI_FLOAT, myID - 1, TAG_DOWN, MPI_COMM_WORLD, &rqRecvDown);
        }
        if(myID < numProcs - 1) {
            MPI_Isend(grid[height], n+2, MPI_FLOAT, myID + 1, TAG_DOWN, MPI_COMM_WORLD, &rqSendDown);
            MPI_Irecv(grid[height+1], n+2, MPI_FLOAT, myID + 1, TAG_UP, MPI_COMM_WORLD, &rqRecvUp);
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


    MPI_Gather(grid[1], (n + 2) * height, MPI_FLOAT, globalGrid, (n + 2) * height, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Calcul du maximum global */
    MPI_Reduce(&globalDiff, &diff, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
 
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
        printf("Utilisation : gauss_seidel <n> <max_iter>\nn : largeur de la grille, supérieure à 3\n max_iter : nombre d'itérations, supérieur à 1");
        exit(EXIT_FAILURE);
    }

    myResult = gauss_seidel(n, maxIter, myID, numProcs);
    
    if(myID == 0) {
        printf("Grid :\n");
        print_buffer(myResult.grid, (n + 2) * n, n + 2);
        printf("Différence maximale = %f\n", myResult.diff);
    }    

    MPI_Finalize(); 
    
    return 0;
}


