
#include "scalarProd.h"

int main()
{
  float *h_a;
  float *h_b;
  float *d_a;

  // Allocate memory in host
  Manage_Memory(0,&h_a,&h_b,&d_a);

  // Allocate memory in device
  Manage_Memory(1,&h_a,&h_b,&d_a);

  // Initialize h_a
  for (int i = 0; i < N; i++) {
    h_a[i] = i;
  }

  // Send data to device
  Manage_Comms(0,&h_a,&d_a);

  // Call GPU function
  My_GPU_Func(&d_a);

  // Brind result to host
  Manage_Comms(1,&h_b,&d_a);

  // Save result to file
  Save_Results(h_b);

  // Free memory 
  Manage_Memory(2,&h_a,&h_b,&d_a);
  
  return 0;
}
