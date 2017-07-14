#include<stdio.h>
#include<cuda.h>
#define Xdim 8
#define Ydim 8
texture<int2,2>myTextureData;

// Compiling as: 
// nvcc -arch=sm_35 doublePrecisionTexMem.cu

static __inline__ __device__ double fetch_double(int2 p){
    return __hiloint2double(p.y, p.x);
}

__global__ void kern(double *o, int pitch){
    __shared__ double A[Xdim][Ydim];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    int2 myData;

    if(i<Xdim && j<Ydim){

        myData = tex2D(myTextureData, i, j);

        A[threadIdx.x][threadIdx.y] = fetch_double(myData);
    }
    __syncthreads();

    if(i<Xdim && j<Ydim){
        o[j*pitch + i] = A[threadIdx.x][threadIdx.y];
    }
}

int main(int argc, char *argv[]){
    double hbuf[Xdim][Ydim];
    double hout[Xdim][Ydim]; 
    double *dob; 
    double *dbuf;
    size_t pitch_bytes;

    cudaMallocPitch((void**)&dbuf,&pitch_bytes,sizeof(double)*Xdim,Ydim);
    cudaMallocPitch((void**)&dob, &pitch_bytes,sizeof(double)*Xdim,Ydim);

    // 8x8 data array
    hbuf[0][0] = 1.234567891234567;
    hbuf[0][1] = 12.34567891234567;
    hbuf[0][2] = 123.4567891234567;
    hbuf[0][3] = 1234.567891234567;
    hbuf[0][4] = 12345.67891234567;
    hbuf[0][5] = 123456.7891234567;
    hbuf[0][6] = 1234567.891234567;
    hbuf[0][7] = 12345678.91234567;
    hbuf[1][0] = 123456789.1234567;
    hbuf[1][1] = 1234567891.234567;
    hbuf[1][2] = 12345678912.34567;
    hbuf[1][3] = 123456789123.4567;
    hbuf[1][4] = 1234567891234.567;
    hbuf[1][5] = 12345678912345.67;
    hbuf[1][6] = 123456789123456.7;
    hbuf[1][7] = 1234567891234567.;
    hbuf[2][0] = 123456789.7654321;
    hbuf[2][1] = 1234567897.654321;
    hbuf[2][2] = 12345678976.54321;
    hbuf[2][3] = 123456789765.4321;
    hbuf[2][4] = 1234567897654.321;
    hbuf[2][5] = 12345678976543.21;
    hbuf[2][6] = 123456789765432.1;
    hbuf[2][7] = 1234567897654321.;
    hbuf[3][0] = 9.876543211234567;
    hbuf[3][1] = 98.76543211234567;
    hbuf[3][2] = 987.6543211234567;
    hbuf[3][3] = 9876.543211234567;
    hbuf[3][4] = 98765.43211234567;
    hbuf[3][5] = 987654.3211234567;
    hbuf[3][6] = 9876543.211234567;
    hbuf[3][7] = 98765432.11234567;
    hbuf[4][0] = 987654321.1234567;
    hbuf[4][1] = 9876543211.234567;
    hbuf[4][2] = 98765432112.34567;
    hbuf[4][3] = 987654321123.4567;
    hbuf[4][4] = 9876543211234.567;
    hbuf[4][5] = 98765432112345.67;
    hbuf[4][6] = 987654321123456.7;
    hbuf[4][7] = 9876543211234567.;
    hbuf[5][0] = 987654321.7654321;
    hbuf[5][1] = 9876543217.654321;
    hbuf[5][2] = 98765432176.54321;
    hbuf[5][3] = 987654321765.4321;
    hbuf[5][4] = 9876543217654.321;
    hbuf[5][5] = 98765432176543.21;
    hbuf[5][6] = 987654321765432.1;
    hbuf[5][7] = 9876543217654321.;
    hbuf[6][0] = 1234567891234567.;
    hbuf[6][1] = 123456789123456.7;
    hbuf[6][2] = 12345678912345.67;
    hbuf[6][3] = 1234567891234.567;
    hbuf[6][4] = 123456789123.4567;
    hbuf[6][5] = 12345678912.34567;
    hbuf[6][6] = 1234567891.234567;
    hbuf[6][7] = 123456789.1234567;
    hbuf[7][0] = 12345678.91234567;
    hbuf[7][1] = 1234567.891234567;
    hbuf[7][2] = 123456.7891234567;
    hbuf[7][3] = 12345.67891234567;
    hbuf[7][4] = 1234.567891234567;
    hbuf[7][5] = 123.4567891234567;
    hbuf[7][6] = 12.34567891234567;
    hbuf[7][7] = 1.234567891234567; 

    // Display the array
    for (int i=0; i<Xdim; i++){
        for(int j=0; j<Ydim; j++){

            printf("%.4E\t", (float)hbuf[i][j]);
        }
        printf("\n");
    }

    cudaMemcpy2D(dbuf, pitch_bytes, hbuf, Xdim*sizeof(double), Xdim*sizeof(double), Ydim, cudaMemcpyHostToDevice);

    myTextureData.addressMode[0] = cudaAddressModeClamp;
    myTextureData.addressMode[1] = cudaAddressModeClamp;
    myTextureData.filterMode = cudaFilterModePoint;
    myTextureData.normalized = false;  

    cudaBindTexture2D(0, myTextureData, dbuf, cudaCreateChannelDesc(32,32,0,0, cudaChannelFormatKindSigned), Xdim, Ydim, pitch_bytes ); 

    int pitch = pitch_bytes/sizeof(double);

    dim3 blockPerGrid(1, 1);
    dim3 threadPerBlock(8, 8);
    kern<<<blockPerGrid, threadPerBlock>>>(dob, pitch);

    cudaMemcpy2D(hout,Xdim*sizeof(double), dob, pitch_bytes, Xdim*sizeof(double), Ydim, cudaMemcpyDeviceToHost);

    printf("\nI am Fine\n\n");

    for(int i=0 ; i<Xdim ; i++){
        for(int j=0; j<Ydim; j++){
            printf("%.4E\t", (float)hout[i][j]);
        }
        printf("\n");
    }

    // Free memory
    cudaUnbindTexture(myTextureData);
    cudaFree(dbuf);
    cudaFree(dob);
    return 0;
}