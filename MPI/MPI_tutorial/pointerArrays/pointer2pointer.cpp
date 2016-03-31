#include <stdio.h>
#include <stdlib.h>

/* Example found on http://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/ */

int main() {

  int r = 3, c = 4, i, j, count;
  
  int **arr = (int **)malloc(r * sizeof(int *));
  for (i=0; i<r; i++)
    arr[i] = (int *)malloc(c * sizeof(int));
  
  // Note that arr[i][j] is same as *(*(arr+i)+j)
  count = 0;
  for (i = 0; i < r; i++)
    for (j = 0; j < c; j++)
      arr[i][j] = ++count; // OR *(*(arr+i)+j) = ++count
  
  // print array data
  for (i = 0; i < r; i++){
    for (j = 0; j < c; j++){
      printf("%d ", arr[i][j]);
    }
    printf("\n");
  }
  /* Code for further processing and free the dynamically allocated memory */
  
  return 0;
}
