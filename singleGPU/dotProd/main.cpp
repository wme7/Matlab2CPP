
#include "dotProd.h"

int main()
{
  float *h_a;
  float *h_b;
  float *h_c;
  float *d_a;
  float *d_b;
  float *d_c;
  float dot = 0.0;

  // Allocate memory in host
  Manage_Memory(0,&h_a,&h_b,&h_c,&d_a,&d_b,&d_c);

  // Allocate memory in device
  Manage_Memory(1,&h_a,&h_b,&h_c,&d_a,&d_b,&d_c);

  // Initialize h_a and h_b
  for (int i = 0; i < N; i++) {
    h_a[i] = 1;
    h_b[i] = 2;
  }

  // Send data to device
  Manage_Comms(0,&h_a,&h_b,&h_c,&d_a,&d_b,&d_c);

  // Call GPU function
  My_GPU_Func(&d_a,&d_b,&d_c);

  // Brind result to host
  Manage_Comms(1,&h_a,&h_b,&h_c,&d_a,&d_b,&d_c);

  // get reduction result
  for (int i = 0; i < BLOCKS; i++) {
    dot += h_c[i];
  }

  // output result
  printf("Computation successful, ( a * b ) = %g \n",dot);

  // Free memory 
  Manage_Memory(2,&h_a,&h_b,&h_c,&d_a,&d_b,&d_c);
  
  return 0;
}
