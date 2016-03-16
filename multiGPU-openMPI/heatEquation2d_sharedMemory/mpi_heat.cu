#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define USE_CPU 0
#define USE_GPU 1

// macro for checking for errors in kernel launches
#  define CHECK_ERROR(errorMessage) do {                                     \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = cudaThreadSynchronize();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


// DIVIDE_INTO(x/y) for integers, used to determine # of blocks/warps etc.
#define DIVIDE_INTO(x,y) (((x) + (y) - 1)/(y))

// I2D to index into a linear memory space from a 2D array index pair
#define I2D(ni, i, j) ((i) + (ni)*(j))

// macro for making CUDA grids
dim3 make_large_grid(const unsigned int num_threads, const unsigned int blocksize){
    const unsigned int num_blocks = DIVIDE_INTO(num_threads, blocksize);
    if (num_blocks <= 65535){
        //fits in a 1D grid
        return dim3(num_blocks);
    } else {
        //2D grid is required
        const unsigned int side = (unsigned int) ceil(sqrt((double)num_blocks));
        return dim3(side,side);
    }
}
// macro for getting CUDA thread ids
#define large_grid_thread_id(void) ((__umul24(blockDim.x,blockIdx.x + __umul24(blockIdx.y,gridDim.x)) + threadIdx.x))


// Block size in the i and j directions
#define NI_TILE 64
#define NJ_TILE 8


static void gather_cpu(int ni, int nj, float *temp, float *buf_left,
                       float *buf_right) {
    
    // gathers from the temperature array into message buffers
    static int *list_left, *list_right;
    static int done_init = -1;
    
    int i, mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (done_init == -1) {
        list_left = (int *)malloc(sizeof(int)*(nj-2));
        list_right = (int *)malloc(sizeof(int)*(nj-2));
        for (i=0; i < nj-2; i++) {
            list_left[i] = I2D(ni, 1, i+1);
            list_right[i] = I2D(ni, ni-2, i+1);
        }
        done_init = 1;
    }
    for (i=0; i < nj-2; i++) {
        buf_left[i] = temp[list_left[i]];
        buf_right[i] = temp[list_right[i]];
    }
}

static void scatter_cpu(int ni, int nj, float *temp, float *buf_left,
                        float *buf_right) {

    // scatters from message buffers info temperature array
    static int *list_left, *list_right;
    static int done_init = -1;
    int i, mpi_rank, mpi_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (done_init == -1) {
        list_left = (int *)malloc(sizeof(int)*(nj-2));
        list_right = (int *)malloc(sizeof(int)*(nj-2));
        for (i=0; i < nj-2; i++) {
            list_left[i] = I2D(ni, 0, i+1);
            list_right[i] = I2D(ni, ni-1, i+1);
        }
        done_init = 1;
    }
    for (i=0; i < nj-2; i++) {
        if (mpi_rank == 0) {
            temp[list_right[i]] = buf_right[i];
        }
        else if (mpi_rank > 0 && mpi_rank < mpi_size-1) {
            temp[list_left[i]] = buf_left[i];
            temp[list_right[i]] = buf_right[i];
        }
        else {
            temp[list_left[i]] = buf_left[i];
        }
    }
}
__global__ static void gather_gpu_kernel(int n, float *buf, int *list, float *temp) {
    // gather kernel for gpu
    int i;
    i = large_grid_thread_id();
    if (i < n) {
        buf[i] = temp[list[i]];
    }
}

__global__ static void scatter_gpu_kernel(int n, float *buf, int *list, float *temp) {
    // scatter kernel for gpu
    int i;
    i = large_grid_thread_id();
    if (i < n) {
        temp[list[i]] = buf[i];
    }
}

static void gather_gpu(int ni, int nj, float *temp_d, 
                       float *buf_left_h,
                       float *buf_right_h,
                       float *buf_left_d,
                       float *buf_right_d) {
    
    // gathers from the temperature array into message buffers

    static int *list_left_h, *list_right_h;
    static int *list_left_d, *list_right_d;
    static int done_init = -1;
    dim3 grid_dim, block_dim;
    
    int i, mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // initialise gather lists
    if (done_init == -1) {
        list_left_h = (int *)malloc(sizeof(int)*(nj-2));
        list_right_h = (int *)malloc(sizeof(int)*(nj-2));
        cudaMalloc((void **)&list_left_d, sizeof(int)*(nj-2));
        cudaMalloc((void **)&list_right_d, sizeof(int)*(nj-2));
        for (i=0; i < nj-2; i++) {
            list_left_h[i] = I2D(ni, 1, i+1);
            list_right_h[i] = I2D(ni, ni-2, i+1);
        }
        cudaMemcpy((void *)list_left_d, (void *)list_left_h, sizeof(float)*(nj-2),
               cudaMemcpyHostToDevice);
        cudaMemcpy((void *)list_right_d, (void *)list_right_h, sizeof(float)*(nj-2),
               cudaMemcpyHostToDevice);
        done_init = 1;
    }
    
    grid_dim = make_large_grid(nj-2, 16);
    block_dim = dim3(16);

    // gather on gpu
    gather_gpu_kernel<<<grid_dim, block_dim>>>(nj-2, buf_left_d, list_left_d, temp_d);
    gather_gpu_kernel<<<grid_dim, block_dim>>>(nj-2, buf_right_d, list_right_d, temp_d);
    CHECK_ERROR("kernel failed\n");
    // copy buffers to cpu
    cudaMemcpy((void *)buf_left_h, (void *)buf_left_d, sizeof(float)*(nj-2),
               cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)buf_right_h, (void *)buf_right_d, sizeof(float)*(nj-2),
               cudaMemcpyDeviceToHost);

}

static void scatter_gpu(int ni, int nj, float *temp_d, 
                        float *buf_left_h,
                        float *buf_right_h,
                        float *buf_left_d,
                        float *buf_right_d) {

    // scatters from message buffers info temperature array

    static int *list_left_h, *list_right_h;
    static int *list_left_d, *list_right_d;
    static int done_init = -1;
    int i, mpi_rank, mpi_size;
    dim3 grid_dim, block_dim;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // initialise gather lists
    if (done_init == -1) {
        list_left_h = (int *)malloc(sizeof(int)*(nj-2));
        list_right_h = (int *)malloc(sizeof(int)*(nj-2));
        cudaMalloc((void **)&list_left_d, sizeof(int)*(nj-2));
        cudaMalloc((void **)&list_right_d, sizeof(int)*(nj-2));
        
        for (i=0; i < nj-2; i++) {
            list_left_h[i] = I2D(ni, 0, i+1);
            list_right_h[i] = I2D(ni, ni-1, i+1);
        }

        cudaMemcpy((void *)list_left_d, (void *)list_left_h, sizeof(float)*(nj-2),
               cudaMemcpyHostToDevice);
        cudaMemcpy((void *)list_right_d, (void *)list_right_h, sizeof(float)*(nj-2),
               cudaMemcpyHostToDevice);
        
        done_init = 1;
    }

    // copy buffers to gpu
    cudaMemcpy((void *)buf_left_d, (void *)buf_left_h, sizeof(float)*(nj-2),
               cudaMemcpyHostToDevice);
    cudaMemcpy((void *)buf_right_d, (void *)buf_right_h, sizeof(float)*(nj-2),
               cudaMemcpyHostToDevice);


    // scatter on gpu
    grid_dim = make_large_grid(nj-2, 16);
    block_dim = dim3(16);
    if (mpi_rank == 0) {
        scatter_gpu_kernel<<<grid_dim, block_dim>>>(nj-2, buf_right_d, list_right_d, temp_d);
    }
    else if (mpi_rank > 0 && mpi_rank < mpi_size-1) {
        scatter_gpu_kernel<<<grid_dim, block_dim>>>(nj-2, buf_left_d, list_left_d, temp_d);
        scatter_gpu_kernel<<<grid_dim, block_dim>>>(nj-2, buf_right_d, list_right_d, temp_d);
    }
    else {
        scatter_gpu_kernel<<<grid_dim, block_dim>>>(nj-2, buf_left_d, list_left_d, temp_d);
    }
    CHECK_ERROR("kernel failed\n");
}



// kernel to update temperatures - CPU version
void step_kernel_cpu(int ni, 
                     int nj,
                     float tfac, 
                     float *temp_in,
                     float *temp_out) {
    int i, j, i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;

    // loop over all points in domain (not boundary points)
    for (j=1; j < nj-1; j++) {
        for (i=1; i < ni-1; i++) {
            // find indices into linear memory for central point and neighbours
            i00 = I2D(ni, i, j);
            im10 = I2D(ni, i-1, j);
            ip10 = I2D(ni, i+1, j);
            i0m1 = I2D(ni, i, j-1);
            i0p1 = I2D(ni, i, j+1);

            // evaluate derivatives
            d2tdx2 = temp_in[im10] - 2.0f*temp_in[i00] + temp_in[ip10];
            d2tdy2 = temp_in[i0m1] - 2.0f*temp_in[i00] + temp_in[i0p1];
            
            // update temperatures
            temp_out[i00] = temp_in[i00] + tfac*(d2tdx2 + d2tdy2);
        }
    }
}

// kernel to update temperatures - GPU version (using shared mem)
__global__ void step_kernel_gpu_shared(int ni, 
                                       int nj,
                                       float tfac,
                                       float *temp_in,
                                       float *temp_out) 
{
    int i, j, ti, tj, i00, i0m1, i0p1, ntot, j_iter;
    int j_sh, jm1_sh, jp1_sh, tmp_sh;

    float d2tdx2, d2tdy2;
    bool compute_i;

    // allocate an array in shared memory
    __shared__ float temp[NI_TILE][3];
    
    // find i and j indices of current thread
    ti = threadIdx.x;
    tj = threadIdx.y;
    i = blockIdx.x*(NI_TILE-2) + ti;
    j = blockIdx.y*(NJ_TILE) + tj + 1;

    // indices into linear memory for current thread
    i00 = i + ni*j;
    i0m1 = i + ni*(j-1);
    i0p1 = i + ni*(j+1);

    // initial shared memory planes
    jm1_sh = 0;
    j_sh = 1;
    jp1_sh = 2;
   
    
    // read the first two planes from global memory into shared memory
    // (if thread is not outside domain)
    ntot = ni*nj;
    if (i00 < ntot) {
        temp[ti][jm1_sh] = temp_in[i0m1];
        temp[ti][j_sh] = temp_in[i00];
    }
    
    // make sure all threads have read in data
    __syncthreads();


    compute_i = i > 0 && i < ni-1 && ti > 0 && ti < NI_TILE-1;

    for (j_iter=0; j_iter < NJ_TILE; j_iter++) {
        
        // read in the next plane
        if (i0p1 < ntot) {
            temp[ti][jp1_sh] = temp_in[i0p1];
        }
        __syncthreads();

        // only compute if (a) thread is within the domain
        // and (b) thread is not on boundary of a thread block
        if (compute_i && j < nj-1) {
                
            // evaluate derivatives
            d2tdx2 = (temp[ti+1][j_sh] - 2.0f*temp[ti][j_sh] + temp[ti-1][j_sh]);
            d2tdy2 = (temp[ti][jp1_sh] - 2.0f*temp[ti][j_sh] + temp[ti][jm1_sh]);
            
            // update temperatures
            temp_out[i00] = temp_in[i00] + tfac*(d2tdx2 + d2tdy2);
        }
        __syncthreads();
        
        i00 += ni;
        i0m1 += ni;
        i0p1 += ni;
        j+= 1;
        
        // swap shared memory planes
        tmp_sh = jm1_sh;
        jm1_sh = j_sh;
        j_sh = jp1_sh;
        jp1_sh = tmp_sh;
        
    }
}

int main(int argc, char *argv[]) {

    
    float *temp1_h, *temp2_h,  *temp1_d, *temp2_d, *temp_tmp;
    float *buf_left_in_h, *buf_right_in_h, *buf_left_out_h, *buf_right_out_h;
    float *buf_left_in_d, *buf_right_in_d, *buf_left_out_d, *buf_right_out_d;
    
    int ni, nj, nstep;
    int i, j, i2d, istep, ist, ien;
    float temp_bl, temp_br, temp_tl, temp_tr, tfac; 
    dim3 grid_dim, block_dim;
    FILE *fp;
    char fname[200];

    MPI_Request req_in[2], req_out[2];
    MPI_Status stat_in[2], stat_out[2];

    int mpi_size, mpi_rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // domain size and number of timesteps (iterations)
    ni = 2048;
    nj = 512;
    nstep = 100;
    
    // allocate temperature array on host
    temp1_h = (float *)malloc(sizeof(float)*ni*nj);
    temp2_h = (float *)malloc(sizeof(float)*ni*nj);

    // allocate buffer arrays on host
    buf_left_in_h = (float *)malloc(sizeof(float)*(nj-2));
    buf_right_in_h = (float *)malloc(sizeof(float)*(nj-2));
    buf_left_out_h = (float *)malloc(sizeof(float)*(nj-2));
    buf_right_out_h = (float *)malloc(sizeof(float)*(nj-2));

    // allocate temperature arrays on device
    cudaMalloc((void **)&temp1_d, sizeof(float)*ni*nj);
    cudaMalloc((void **)&temp2_d, sizeof(float)*ni*nj);

    // allocate buffer arrays on device
    cudaMalloc((void **)&buf_left_in_d, sizeof(float)*(nj-2));
    cudaMalloc((void **)&buf_right_in_d, sizeof(float)*(nj-2));
    cudaMalloc((void **)&buf_left_out_d, sizeof(float)*(nj-2));
    cudaMalloc((void **)&buf_right_out_d, sizeof(float)*(nj-2));


    // initial temperature in interior
    for (j=1; j < nj-1; j++) {
        for (i=1; i < ni-1; i++) {
            i2d = i + ni*j;
            temp1_h[i2d] = 0.0f;
        }
    }
    
    // initial temperature on boundaries - set corners
    temp_bl = 200.0f;
    temp_br = 200.0f;
    temp_tl = 200.0f;
    temp_tr = 200.0f;

    // set edges by linear interpolation from corners
    for (i=0; i < ni; i++) {
        // bottom
        j = 0;
        i2d = i + ni*j;
        temp1_h[i2d] = temp_bl + (temp_br-temp_bl)*(float)i/(float)(ni-1);

        // top
        j = nj-1;
        i2d = i + ni*j;
        temp1_h[i2d] = temp_tl + (temp_tr-temp_tl)*(float)i/(float)(ni-1);
    }

    for (j=0; j < nj; j++) {
        // left
        i = 0;
        i2d = i + ni*j;
        temp1_h[i2d] = temp_bl + (temp_tl-temp_bl)*(float)j/(float)(nj-1);

        // right
        i = ni-1;
        i2d = i + ni*j;
        temp1_h[i2d] = temp_br + (temp_tr-temp_br)*(float)j/(float)(nj-1);
    }

    // duplicate temeperature array on host
    memcpy(temp2_h, temp1_h, sizeof(float)*ni*nj);
        
    

    // transfer temperature array from host to device
    cudaMemcpy((void *)temp1_d, (void *)temp1_h, sizeof(float)*ni*nj,
               cudaMemcpyHostToDevice);
    cudaMemcpy((void *)temp2_d, (void *)temp1_h, sizeof(float)*ni*nj,
               cudaMemcpyHostToDevice);
    

    tfac = 0.2f;

    // main iteration loop   
    for (istep=0; istep < nstep; istep++) {


        // POST RECEIVES
        // rank 0 only receives from right
        if (mpi_rank == 0) {
            MPI_Irecv(buf_right_in_h, nj-2, MPI_FLOAT, mpi_rank+1, 
                      2*mpi_rank+1, MPI_COMM_WORLD, &req_in[0]);
        }
        // middle ranks receive both left and right
        else if (mpi_rank > 0 && mpi_rank < mpi_size-1) {
            MPI_Irecv(buf_left_in_h, nj-2, MPI_FLOAT, mpi_rank-1, 
                      2*mpi_rank, MPI_COMM_WORLD, &req_in[0]);
            MPI_Irecv(buf_right_in_h, nj-2, MPI_FLOAT, mpi_rank+1, 
                      2*mpi_rank+1, MPI_COMM_WORLD, &req_in[1]);
        }
        // rank mpi_size-1 receives from left
        else {
            MPI_Irecv(buf_left_in_h, nj-2, MPI_FLOAT, mpi_rank-1, 
                      2*mpi_rank, MPI_COMM_WORLD, &req_in[0]);
        }


        // GATHER INTO OUTGOING BUFFERS
        if (USE_CPU == 1) {
            gather_cpu(ni, nj, temp1_h, buf_left_out_h, buf_right_out_h);
        }
        else {
            
            gather_gpu(ni, nj, temp1_d, buf_left_out_h, buf_right_out_h,
                       buf_left_out_d, buf_right_out_d);
            
        }

        
        // POST SENDS
        // rank 0 only sends to the right
        if (mpi_rank == 0) {
            MPI_Isend(buf_right_out_h, nj-2, MPI_FLOAT, mpi_rank+1, 
                      2*(mpi_rank+1), MPI_COMM_WORLD, &req_out[0]);
        }
        // middle ranks send both left and right
        else if (mpi_rank > 0 && mpi_rank < mpi_size-1) {
            MPI_Isend(buf_left_out_h, nj-2, MPI_FLOAT, mpi_rank-1, 
                      2*(mpi_rank-1)+1, MPI_COMM_WORLD, &req_out[0]);
            MPI_Isend(buf_right_out_h, nj-2, MPI_FLOAT, mpi_rank+1, 
                      2*(mpi_rank+1), MPI_COMM_WORLD, &req_out[1]);
        }
        // rank mpi_size-1 sends to the left
        else {
            MPI_Isend(buf_left_out_h, nj-2, MPI_FLOAT, mpi_rank-1, 
                      2*(mpi_rank-1)+1, MPI_COMM_WORLD, &req_out[0]);
        }


        // SCATTER INTO TEMPERATURE ARRAYS FROM INCOMING BUFFERS 
        if (USE_CPU == 1) {
            scatter_cpu(ni, nj, temp1_h, buf_left_in_h, buf_right_in_h);
        }
        else {
            scatter_gpu(ni, nj, temp1_d, buf_left_in_h, buf_right_in_h,
                        buf_left_in_d, buf_right_in_d);
            
        }


        // WAIT FOR MPI COMMUNICATION TO FINISH
        if (mpi_rank == 0) {
            MPI_Waitall(1, &req_in[0], &stat_in[0]);
            MPI_Waitall(1, &req_out[0], &stat_out[0]);
        }
        else if (mpi_rank > 0 && mpi_rank < mpi_size-1) {
            MPI_Waitall(2, req_in, stat_in);
            MPI_Waitall(2, req_out, stat_out);
        }
        else {
            MPI_Waitall(1, &req_in[0], &stat_in[0]);
            MPI_Waitall(1, &req_out[0], &stat_out[0]);
        }

        // EXECUTE KERNEL
        if (USE_CPU == 1) {
            step_kernel_cpu(ni, nj, tfac, temp1_h, temp2_h);
            temp_tmp = temp1_h;
            temp1_h = temp2_h;
            temp2_h = temp_tmp;
        }
        else {
            grid_dim = dim3(DIVIDE_INTO(ni-2,NI_TILE-2), DIVIDE_INTO(nj-2,NJ_TILE), 1);
            block_dim = dim3(NI_TILE, 1, 1);
	    
            // launch kernel
            step_kernel_gpu_shared<<<grid_dim, block_dim>>>(ni, nj,
                                                            tfac, temp1_d, temp2_d);
            CHECK_ERROR("kernel failed\n");
            
            // swap the temp pointers
            temp_tmp = temp1_d;
            temp1_d = temp2_d;
            temp2_d = temp_tmp;
        }
    }

    // copy temperature array from device to host
    if (USE_CPU == 0) {
        cudaMemcpy((void *)temp1_h, (void *)temp1_d, sizeof(float)*ni*nj,
                   cudaMemcpyDeviceToHost);
    }
    
    // output temperature to files - one for each process
    if (mpi_rank == 0) {
        ist = 0; 
        ien = ni;
    }
    else if (mpi_rank == 1 && mpi_rank < mpi_size-1) {
        ist = 1;
        ien = ni-1;
    }
    else {
        ist = 1;
        ien = ni;
    }
    sprintf(fname, "out%i.dat", mpi_rank);
    fp = fopen(fname, "w");
    for (j=0; j < nj; j++) {
        for (i=ist; i < ien; i++) {
            fprintf(fp, "%f\n", i, j, temp1_h[i + ni*j]);
        }
    }
    fclose(fp);
        
}
