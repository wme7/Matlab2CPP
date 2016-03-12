#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

// set USE_CPU to 1 to run on CPU
// set USE_GPU to 1 (and USE_CPU to 0) to use GPU kernel - without shared mem
// set USE_GPU to 2 (and USE_CPU to 0) to use GPU kernel - with shared mem
// set USE_GPU to 3 (and USE_CPU to 0) to use GPU kernel - with shared mem_v2

#define USE_CPU 0
#define USE_GPU 3

// DIVIDE_INTO(x/y) for integers, used to determine # of blocks/warps etc.
#define DIVIDE_INTO(x,y) (((x) + (y) - 1)/(y))
// I2D to index into a linear memory space from a 2D array index pair
#define I2D(ni, i, j) ((i) + (ni)*(j))

// Block size in the i and j directions
#define NI_TILE 64
#define NJ_TILE 8


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
            d2tdx2 = temp_in[im10] - 2*temp_in[i00] + temp_in[ip10];
            d2tdy2 = temp_in[i0m1] - 2*temp_in[i00] + temp_in[i0p1];

	    // update temperatures
            temp_out[i00] = temp_in[i00] + tfac*(d2tdx2 + d2tdy2);
        }
    }
}

// kernel to update temperatures - GPU version (not using shared mem)
__global__ void step_kernel_gpu(int ni, 
                                int nj,
                                float tfac,
                                float *temp_in,
                                float *temp_out) 
{
    int i, j, ti, tj, i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;

    // find i and j indices of this thread
    ti = threadIdx.x;
    tj = threadIdx.y;
    i = blockIdx.x*(NI_TILE) + ti;
    j = blockIdx.y*(NJ_TILE) + tj;

    // find indices into linear memory 
    i00 = I2D(ni, i, j);
    im10 = I2D(ni, i-1, j);
    ip10 = I2D(ni, i+1, j);
    i0m1 = I2D(ni, i, j-1);
    i0p1 = I2D(ni, i, j+1);
    
    // check that thread is within domain (not on boundary or outside domain)
    if (i > 0 && i < ni-1 && j > 0 && j < nj-1) {
            // evaluate derivatives 
            d2tdx2 = temp_in[im10] - 2*temp_in[i00] + temp_in[ip10];
            d2tdy2 = temp_in[i0m1] - 2*temp_in[i00] + temp_in[i0p1];
   	             
            // update temperature
            temp_out[i00] = temp_in[i00] + tfac*(d2tdx2 + d2tdy2);
        
    }
}

// kernel to update temperatures - GPU version (using shared mem)
__global__ void step_kernel_gpu_shared(int ni, 
                                       int nj,
                                       float tfac,
                                       float *temp_in,
                                       float *temp_out) 
{
    int i, j, ti, tj, i2d;
    // allocate an array in shared memory
    __shared__ float temp[NI_TILE][NJ_TILE];
    float d2tdx2, d2tdy2;
    
    // find i and j indices of current thread
    ti = threadIdx.x;
    tj = threadIdx.y;
    i = blockIdx.x*(NI_TILE-2) + ti;
    j = blockIdx.y*(NJ_TILE-2) + tj;

    // index into linear memory for current thread
    i2d = i + ni*j;
    
    // read from global memory into shared memory
    //  (if thread is not outside domain)
        if (i2d < ni*nj) {
        temp[ti][tj] = temp_in[i2d];
	}

    // make sure all threads have read in data
    __syncthreads();

    // only compute if (a) thread is within the whole domain
    if (i > 0 && i < ni-1 && j > 0 && j < nj-1) {
        // and (b) thread is not on boundary of a block
        if ((ti > 0) && (ti < NI_TILE-1) &&
            (tj > 0) && (tj < NJ_TILE-1)) {
            
            //evaluate derivatives
            d2tdx2 = (temp[ti+1][tj] - 2*temp[ti][tj] + temp[ti-1][tj]);
            d2tdy2 = (temp[ti][tj+1] - 2*temp[ti][tj] + temp[ti][tj-1]);
            
	    // update temperatures
            temp_out[i2d] = temp_in[i2d] + tfac*(d2tdx2 + d2tdy2);
        }
    }
}


// kernel to update temperatures - GPU version (using shared mem_v2)
__global__ void step_kernel_gpu_shared_v2(int ni, 
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



int main(int argc, char *argv[]) 
{
    int ni, nj, nstep;
    float tfac, *temp1_h, *temp2_h,  *temp1_d, *temp2_d, *temp_tmp;
    int i, j, i2d, istep;
    float temp_bl, temp_br, temp_tl, temp_tr;
    dim3 grid_dim, block_dim;
    clock_t startclock, stopclock;
    double timeperstep;
    FILE *fp;
   
    // domain size and number of timesteps (iterations)
    ni = 1024;
    nj = 1024;
    nstep = 10000;
    
    // allocate temperature array on host
    temp1_h = (float *)malloc(sizeof(float)*ni*nj);
    temp2_h = (float *)malloc(sizeof(float)*ni*nj);

    // initial temperature in interior
    for (j=1; j < nj-1; j++) {
        for (i=1; i < ni-1; i++) {
            i2d = i + ni*j;
            temp1_h[i2d] = 0.0f;
        }
    }
    
    // initial temperature on boundaries - set corners
    temp_bl = 200.0f;
    temp_br = 300.0f;
    temp_tl = 200.0f;
    temp_tr = 300.0f;

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
        
    // allocate temperature arrays on device
    cudaMalloc((void **)&temp1_d, sizeof(float)*ni*nj);
    cudaMalloc((void **)&temp2_d, sizeof(float)*ni*nj);

    // transfer temperature array from host to device
    cudaMemcpy((void *)temp1_d, (void *)temp1_h, sizeof(float)*ni*nj,
               cudaMemcpyHostToDevice);
    cudaMemcpy((void *)temp2_d, (void *)temp1_h, sizeof(float)*ni*nj,
               cudaMemcpyHostToDevice);
    

    tfac = 0.2f;
    
    startclock = clock();

    // main iteration loop   
    for (istep=0; istep < nstep; istep++) {
        //printf("%i\n", istep);
        if (USE_CPU == 1) {
            // CPU kernel
            step_kernel_cpu(ni, nj, tfac, temp1_h, temp2_h);
	    // swap the temp pointers
            temp_tmp = temp1_h;
            temp1_h = temp2_h;
            temp2_h = temp_tmp;
            
        }
        if (USE_GPU == 1) {
	    // GPU - no shared memory
	    // set threads and blocks
            grid_dim = dim3(DIVIDE_INTO(ni,NI_TILE), DIVIDE_INTO(nj,NJ_TILE), 1);
            block_dim = dim3(NI_TILE, NJ_TILE, 1);
           
            
            //printf("Step %i\n", istep);
	    // launch kernel 
            step_kernel_gpu<<<grid_dim, block_dim>>>(ni, nj,
                                                     tfac, temp1_d, temp2_d);
            // swap the temp pointers						     
            temp_tmp = temp1_d;
            temp1_d = temp2_d;
            temp2_d = temp_tmp;
        }
        if (USE_GPU == 2) {
	    // GPU - shared memory
	    // set threads and blocks
            // need to compute for n-2 nodes
            // for each N_TILE threads, N_TILE-2 compute
            // number of blocks in each dimension is (n-2)/(N_TILE-2), rounded upwards
            grid_dim = dim3(DIVIDE_INTO(ni-2,NI_TILE-2), DIVIDE_INTO(nj-2,NJ_TILE-2), 1);
            block_dim = dim3(NI_TILE, NJ_TILE, 1);
	    // launch kernel
            step_kernel_gpu_shared<<<grid_dim, block_dim>>>(ni, nj,
                                                            tfac, temp1_d, temp2_d);
            // swap the temp pointers
            temp_tmp = temp1_d;
            temp1_d = temp2_d;
            temp2_d = temp_tmp;
        }
         if (USE_GPU == 3) {
	    // GPU - shared memory - iterate upwards through block using a line of threads
	    // set threads and blocks
            // need to compute for n-2 nodes
            // for each NI_TILE threads, NI_TILE-2 compute
            // number of blocks in each dimension is (n-2)/(N_TILE-2), rounded upwards
            grid_dim = dim3(DIVIDE_INTO(ni-2,NI_TILE-2), DIVIDE_INTO(nj-2,NJ_TILE), 1);
            block_dim = dim3(NI_TILE, 1, 1);
	    // launch kernel
            step_kernel_gpu_shared_v2<<<grid_dim, block_dim>>>(ni, nj,
                                                            tfac, temp1_d, temp2_d);
            // swap the temp pointers
            temp_tmp = temp1_d;
            temp1_d = temp2_d;
            temp2_d = temp_tmp;
        }
    } 

    cudaThreadSynchronize();
    stopclock = clock();
    timeperstep =((double)(stopclock-startclock))/CLOCKS_PER_SEC;
    timeperstep = timeperstep / nstep;
    timeperstep = timeperstep / (ni*nj);

    printf("Time per point per step = %e\n",timeperstep);


    // copy temperature array from device to host
    if (USE_CPU == 0) {
        cudaMemcpy((void *)temp1_h, (void *)temp1_d, sizeof(float)*ni*nj,
                   cudaMemcpyDeviceToHost);
    }
    
    // output temp1 to a file
    FILE *pFile = fopen("result.txt", "w");
    if (pFile != NULL) {
      for (int j = 0; j < nj; j++) {
        for (int i = 0; i < ni; i++) {      
	  fprintf(pFile, "%d\t %d\t %g\n",j,i,temp1_h[i + ni*j]);
        }
      }
      fclose(pFile);
    } else {
    printf("Unable to save to file\n");
    }
}


