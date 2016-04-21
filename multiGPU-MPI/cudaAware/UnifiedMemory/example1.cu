#include <stdio.h>

__global__ void printValue( int *value) {
printf("value %d\n",value[0]);
printf("value %d\n",value[1]);
}
 
void hostFunction(){
int *value;
cudaMallocManaged(&value, 2 * sizeof(int));
value[0]=1;
value[1]=2;
printValue<<< 1, 1 >>>(value);
cudaDeviceSynchronize();
cudaFree(value);
}
 
int main() {
hostFunction();
return 0;
}
