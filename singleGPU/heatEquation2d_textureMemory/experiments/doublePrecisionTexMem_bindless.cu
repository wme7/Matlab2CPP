// Take From 
// https://stackoverflow.com/questions/35137213/texture-objects-for-doubles

#include <vector>
#include <cstdio>

static __inline__ __device__ double fetch_double(uint2 p){
    return __hiloint2double(p.y, p.x);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void my_print(cudaTextureObject_t texObject)
{
    uint2 rval = tex1Dfetch<uint2>(texObject, 0);
    double dval = fetch_double(rval);
    printf("%f\n", dval);
}

int main()
{

    double i = 0.35;
    int numel = 50;

    std::vector<double> h_data(numel, i);
    double* d_data;
    cudaMalloc(&d_data,numel*sizeof(double));
    cudaMemcpy((void*)d_data, &h_data[0], numel*sizeof(double), cudaMemcpyHostToDevice);


    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;


    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_data;
    resDesc.res.linear.sizeInBytes = numel*sizeof(double);
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.desc.y = 32;

    cudaTextureObject_t texObject;
    gpuErrchk(cudaCreateTextureObject(&texObject, &resDesc, &td, NULL));

    my_print<<<1,1>>>(texObject);

    gpuErrchk(cudaDeviceSynchronize());
    return 0;
}