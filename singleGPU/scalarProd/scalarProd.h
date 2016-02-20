
#include <stdlib.h>
#include <stdio.h>

#define DEBUG 1 // show debug messages
#define N 1000 // number of elements

void Manage_Memory(int phase, float **h_a, float **h_b, float **d_a);
void Manage_Comms(int phase, float **h_a, float **d_a);
void My_GPU_Func(float **d_a);
void Save_Results(float *u);
