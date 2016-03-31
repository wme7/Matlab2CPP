#include <stdio.h>
#include <stdlib.h>

/* Example found on http://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/ */
 
int main() {

  int r = 3, c = 4, i, j, count = 0;

  int *arr = (int *)malloc(r * c * sizeof(int));
  
  for (i = 0; i <  r; i++)
    for (j = 0; j < c; j++)
      *(arr + i*c + j) = ++count;
  
  // print array data
  for (i = 0; i <  r; i++) {
    for (j = 0; j < c; j++) {
      printf("%d ", *(arr + i*c + j));
    }
    printf("\n");
  }
  
  /* Code for further processing and free the dynamically allocated memory */
  
  return 0;
}
