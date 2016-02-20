
#include <stdlib.h>
#include <stdio.h>

#define DEBUG 1 // show debug messages
#define N 1000 // number of elements
#define THREADS 128
#define BLOCKS 32 

void Manage_Memory(int phase,float **h_a,float **h_b,float **h_c,float **d_a,float **d_b,float **d_c);
void Manage_Comms(int phase,float **h_a,float **h_b,float **h_c,float **d_a,float **d_b,float **d_c);
void My_GPU_Func(float **d_a,float **d_b,float **d_c);
