#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// utilities and system includes
// CUDA-C includes
#include "cuda.h"
#define BUFSIZE 256
#define TAG 0
 
 
int devCount;
int myid;
int ihavecuda;
int nodes[256];
int nocuda[256];
int deviceselector=0;
 
 
 
int main(int argc, char *argv[])
 {
        char idstr[256];
        char idstr2[256];
        char buff[BUFSIZE];
        int i;
        int numprocs, rank, namelen;
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        freopen("/dev/null", "w", stderr); /* Hide errors from nodes with no CUDA cards */
        MPI_Status stat;
        MPI_Init(&argc,&argv);
        MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Get_processor_name(processor_name, &namelen);
        MPI_Comm_rank(MPI_COMM_WORLD,&myid);
        if (myid == 0)
        {
                printf("  We have %d processors\n", numprocs);
                printf("  Spawning from %s \n", processor_name);
                printf("  CUDA MPI\n");
                printf("\n");
                for(i=1; i<numprocs;i++)
                {
                        buff[0]='\0';
                        MPI_Send(buff, BUFSIZE, MPI_CHAR, i, TAG, MPI_COMM_WORLD);
                }
                printf("\n\n\n");
                printf("  Probing nodes...\n");
                printf("     Node        Psid  CUDA Cards (devID)\n");
                printf("     ----------- ----- ---- ----------\n");
                for(i=1; i<numprocs;i++)
                {
                        MPI_Recv(buff, BUFSIZE, MPI_CHAR, i, TAG, MPI_COMM_WORLD, &stat);
                        printf("%s\n", buff);
                }
                printf("\n");
                MPI_Finalize(); 
        }
        else
        {
                MPI_Recv(buff, BUFSIZE, MPI_CHAR, 0, TAG, MPI_COMM_WORLD, &stat);
                MPI_Get_processor_name(processor_name, &namelen);
                cudaGetDeviceCount(&devCount);
                buff[0]='\0';
                idstr[0]='\0';
                if (devCount == 0) {
                        sprintf(idstr,"- %-11s %5d %4d NONE", processor_name, rank, devCount);
                        ihavecuda=0;
                }else{
                        ihavecuda=1;
                        if (devCount >= 1){
                                sprintf(idstr, "+ %-11s %5d %4d", processor_name, rank, devCount);
                                idstr2[0]='\0';
                                for (int i = 0; i < devCount; ++i)
                                {
                                        cudaDeviceProp devProp;
                                        cudaGetDeviceProperties(&devProp, i);
                                        sprintf(idstr2, " %s (%d) ", devProp.name, i);
                                        strncat(idstr,idstr2,BUFSIZE);
                                }
                        }
                        else
                        {
                                        cudaDeviceProp devProp;
                                        cudaGetDeviceProperties(&devProp, i);
                                        sprintf(idstr, "%-11s %5d %4d %s", processor_name, rank, devCount, devProp.name);
                        }
                }
                strncat(buff, idstr, BUFSIZE);
                MPI_Send(buff, BUFSIZE, MPI_CHAR, 0, TAG, MPI_COMM_WORLD);
        }
        MPI_Finalize();
        return 0;
}
