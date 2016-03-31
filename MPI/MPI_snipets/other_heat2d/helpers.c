#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define TEMP_INITIALE 0.0
#define TEMP_GAUCHE 10
#define TEMP_DROITE 10
#define TEMP_HAUT 10
#define TEMP_BAS  10

/* Initialise une tranche */
void initialize(float** grid, int n, int height, int numProcs, int myID) {

    for(int i = 0; i < height + 2; i++) {
        for(int j = 0; j < n + 2; j++) {
            if(i == 0 && myID == 0) grid[i][j] = TEMP_HAUT;
            else if(i == height + 1 && myID == numProcs - 1) grid[i][j] = TEMP_BAS;
            else if(j == 0) grid[i][j] = TEMP_GAUCHE;
            else if(j == n + 1) grid[i][j] = TEMP_DROITE;
            else grid[i][j] = TEMP_INITIALE;
        }
    }
}

/* Affiche une tranche */
void print(float** grid, int n, int height, int myID) {
    printf("Process %d\n", myID);
    
    for(int i = 0; i < height + 2; i++) {
        for(int j = 0; j < n + 2; j++) {
            printf("%6.2f\t", grid[i][j]);
        }
        printf("\n");
    }
}

/* Affiche une grille de résultats */
void print_buffer(float* buf, int length, int step) {
    for(int i = 0; i < length; i++) {
        printf("%6.2f\t", buf[i]);
        if((i + 1) % step == 0) printf("\n");
    }
    printf("\n");
}

/* Affiche une erreur de message MPI */
void handle_mpi_error(int ierr) {
    int resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];

    if(ierr != MPI_SUCCESS) {
        MPI_Error_string(ierr,err_buffer,&resultlen);
        printf("Erreur : %d\n", ierr);
        MPI_Finalize();
    }
}

