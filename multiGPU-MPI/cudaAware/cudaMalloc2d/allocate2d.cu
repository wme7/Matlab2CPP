#include <stdio.h>

// From question in stackoverflow.com: 
// http://stackoverflow.com/questions/16905899/cuda-2d-array-indexing-giving-unexpected-results

/*
To compute the correct element index into the pitched array we must (see kernel below):

1. Compute the (virtual) row index from the thread index. We do this by taking integer division of the thread index by the width of each (non-pitched) row (in elements, not bytes).

2. Multiply the row index by the width of each pitched row. The width of each pitched row is given by the pitched parameter, which is in bytes. To convert this pitched byte parameter into a pitched element parameter, we divide by the size of each element. Then by multiplying the quantity by the row index computed in step 1, we have now indexed into the correct row.

3. Compute the (virtual) column index from the thread index by taking the remainder (modulo division) of the thread index divided by the width (in elements). Once we have the column index (in elements) we add it to the start-of-the-correct-row index computed in step 2, to identify the element that this thread will be responsible for.

The above is a fair amount of effort for a relatively straightforward operation, which is one example of why I suggest focusing on basic cuda concepts rather than pitched arrays first. For example I would figure how to handle 1 and 2D thread blocks, and 1 and 2D grids, before tackling pitched arrays. Pitched arrays are a useful performance enhancer for accessing 2D arrays (or 3D arrays) in some instances, but they are by no means necessary to handle multidimensional arrays in CUDA.
*/

/*
cudaError_t cudaMallocPitch(void **devPtr,
                            size_t  *pitch,
                            size_t  width (in bytes),
                            size_t  height
                            )   

cudaError_t cudaMemcpy2D(   void  *dst,
                            size_t  dpitch (Pitch of destination memory),
                            const  void *src,
                            size_t  spitch (Pitch of source memory),
                            size_t  width (columns in bytes),
                            size_t  height (rows),
                            enum cudaMemcpyKind  kind
                            )   

Use of pitched memory may provide somewhat higher global access efficiency, since it aligns the start of each row (or column, depending on storage convention used) for the GPU's wide memory transfers. Whether that matters in this use case is impossible to say without knowing the access pattern and whether the code is memory bound. I would start with a simple flat array and consider use of pitched memory as something to be investigated during the fine-tuning stage of performance optimizations, informed by profiler data.
*/

__global__ void doStuff(float* data, float* result, size_t dpitch, size_t rpitch, int width)
{
    if (threadIdx.x < 9) // take the first 9 threads
    {
        int index = threadIdx.x;
        result[((index/width)*(rpitch/sizeof(float))) + (index%width)] = (float) index;
        //            1                    2                  3        
    }
}

int main(void)
{
    /*
        Setup
    */
    float simple[] = {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0};

    float* data_array;
    float* result_array;

    size_t data_array_pitch, result_array_pitch;
    int height = 3;
    int width = 3;
    int width_in_bytes = width * sizeof(float);

    /*
        Initialize GPU arrays
    */
    cudaMallocPitch(&data_array, &data_array_pitch, width_in_bytes, height);
    cudaMallocPitch(&result_array, &result_array_pitch, width_in_bytes, height);

    /*
        Copy data to GPU
    */
    cudaMemcpy2D(data_array, data_array_pitch, simple, width_in_bytes, width_in_bytes, height, cudaMemcpyHostToDevice);

    /*
        Do stuff
    */
    dim3 threads_per_block(16); dim3 num_blocks(1,1);
    doStuff<<<num_blocks, threads_per_block>>>(data_array, result_array, data_array_pitch, result_array_pitch, width);

    /*
        Get the results
    */
    cudaMemcpy2D(simple, width_in_bytes, result_array, result_array_pitch, width_in_bytes, height, cudaMemcpyDeviceToHost);

    for (int i = 1; i <= 9; ++i)
    {
        printf("%f ", simple[i-1]); 
        if(!(i%3)) printf("\n");
    }
    return 0;
}
