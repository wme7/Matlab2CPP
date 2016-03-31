#include <stdio.h>
#include <stdlib.h>

/* Example found on http://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/ */
  
int main() {

  int r=3, c=4, count=0, i, j; int **arr;
  
  arr  = (int **)malloc(r*sizeof(int *));
  arr[0] = (int *)malloc(c*r*sizeof(int));
  
  for(i = 0; i < r; i++)
    arr[i] = (*arr + c * i);
  
  for (i = 0; i < r; i++)
    for (j = 0; j < c; j++)
      arr[i][j] = ++count;  // OR *(*(arr+i)+j) = ++count
  
  // print array data
  for (i = 0; i <  r; i++) {
    for (j = 0; j < c; j++) {
      printf("%d ", arr[i][j]);
    }
    printf("\n");
  }
  
  return 0;
}
