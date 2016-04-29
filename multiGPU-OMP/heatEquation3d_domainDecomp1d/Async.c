#include "heat3d.h"

#define DEBUG
#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

////////////////////////////////////////////////////////////////////////////////
// A method for checking error in CUDA calls
////////////////////////////////////////////////////////////////////////////////
inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
	#if defined(DEBUG) || defined(_DEBUG)
	if (error != cudaSuccess)
	{
		printf("checkCuda error at %s:%i: %s\n", file, line, 
			cudaGetErrorString(cudaGetLastError()));
		exit(-1);
	}
	#endif

	return;
}

////////////////////////////////////////////////////////////////////////////////
// Program Main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	unsigned int Nx, Ny, Nz, max_iters;
	int blockX, blockY, blockZ;

	if (argc == 8)
	{
		Nx = atoi(argv[1]);
		Ny = atoi(argv[2]);
		Nz = atoi(argv[3]);
		max_iters = atoi(argv[4]);
		blockX = atoi(argv[5]);
		blockY = atoi(argv[6]);
		blockZ = atoi(argv[7]);
	}
	else
	{
		printf("Usage: %s nx ny nz i block_x block_y block_z\n", argv[0]);
		exit(1);
	}

	// Get the number of GPUS
	int number_of_devices;
	checkCuda(cudaGetDeviceCount(&number_of_devices));
  
  if (number_of_devices < 2) {
  	printf("Less than two devices were found.\n");
  	printf("Exiting...\n");

  	return -1;
  }

	// Decompose along the Z-axis
	int _Nz = Nz/number_of_devices;

	// Define constants
	const REAL L = 1.0;
	const REAL h = L/(Nx+1);
	const REAL dt = h*h/6.0;
	const REAL beta = dt/(h*h);
	const REAL c0 = beta;
	const REAL c1 = (1-6*beta);

	// Check if ECC is turned on
	ECCCheck(number_of_devices);

	// Set the number of OpenMP threads
	omp_set_num_threads(10);

	#pragma omp parallel
	{
		unsigned int tid = omp_get_num_threads();

		#pragma omp single
		{
			printf("Number of OpenMP threads: %d\n", tid);
		}
	}

  // CPU memory operations
  int dt_size = sizeof(REAL);

	REAL *u_new, *u_old;

	u_new = (REAL *)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(Nz+2));
	u_old = (REAL *)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(Nz+2));

	init(u_old, u_new, h, Nx, Ny, Nz);

	// Allocate and generate arrays on the host
	size_t pitch_bytes;
	size_t pitch_gc_bytes;

	REAL *h_Unew, *h_Uold, *h_s_Uolds[number_of_devices];
	REAL *h_s_Unews[number_of_devices], *left_send_buffer[number_of_devices];
	REAL *left_receive_buffer[number_of_devices], *right_send_buffer[number_of_devices], *right_receive_buffer[number_of_devices];

	h_Unew = (REAL *)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(Nz+2));
	h_Uold = (REAL *)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(Nz+2));

	init(h_Uold, h_Unew, h, Nx, Ny, Nz);

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		if (tid < number_of_devices) {

			h_s_Unews[tid] = (REAL *)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(_Nz+2));
			h_s_Uolds[tid] = (REAL *)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(_Nz+2));

			right_send_buffer[tid] = (REAL *)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(_GC_DEPTH));
			left_send_buffer[tid] = (REAL *)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(_GC_DEPTH));
			right_receive_buffer[tid] = (REAL *)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(_GC_DEPTH));
			left_receive_buffer[tid] = (REAL *)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(_GC_DEPTH));

			checkCuda(cudaHostAlloc((void**)&h_s_Unews[tid], dt_size*(Nx+2)*(Ny+2)*(_Nz+2), cudaHostAllocPortable));
			checkCuda(cudaHostAlloc((void**)&h_s_Uolds[tid], dt_size*(Nx+2)*(Ny+2)*(_Nz+2), cudaHostAllocPortable));
			checkCuda(cudaHostAlloc((void**)&right_send_buffer[tid], dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));
			checkCuda(cudaHostAlloc((void**)&left_send_buffer[tid], dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));
			checkCuda(cudaHostAlloc((void**)&right_receive_buffer[tid], dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));
			checkCuda(cudaHostAlloc((void**)&left_receive_buffer[tid], dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));

			init_subdomain(h_s_Uolds[tid], h_Uold, Nx, Ny, _Nz, tid);
		}
	}

	cudaStream_t compute_stream_6, compute_stream_7, compute_stream_8, compute_stream_9;

  #pragma omp parallel
  {
    unsigned int tid = omp_get_thread_num();

    if (tid == 6) {
      checkCuda(cudaSetDevice(0));
      checkCuda(cudaStreamCreate(&compute_stream_6));
    }
    if (tid == 7) {
      checkCuda(cudaSetDevice(1));
      checkCuda(cudaStreamCreate(&compute_stream_7));
    }
    if (tid == 8) {
      checkCuda(cudaSetDevice(2));
      checkCuda(cudaStreamCreate(&compute_stream_8));
    }
    if (tid == 9) {
      checkCuda(cudaSetDevice(3));
      checkCuda(cudaStreamCreate(&compute_stream_9));
    }
  }

	// GPU stream operations
	cudaStream_t left_send_stream[number_of_devices], right_send_stream[number_of_devices];
	cudaStream_t right_receive_stream[number_of_devices], left_receive_stream[number_of_devices];

	for (int i = 0; i < number_of_devices; i++)
	{
		checkCuda(cudaSetDevice(i));
		checkCuda(cudaStreamCreate(&right_send_stream[i]));
		checkCuda(cudaStreamCreate(&left_send_stream[i]));
		checkCuda(cudaStreamCreate(&right_receive_stream[i]));
		checkCuda(cudaStreamCreate(&left_receive_stream[i]));
	}

	// GPU memory operations
	REAL *d_s_Unews[number_of_devices], *d_s_Uolds[number_of_devices];
	REAL *d_right_send_buffer[number_of_devices];
	REAL *d_left_send_buffer[number_of_devices], *d_right_receive_buffer[number_of_devices];
	REAL *d_left_receive_buffer[number_of_devices];

#pragma omp parallel
{
	int tid = omp_get_thread_num();

	if (tid < number_of_devices) {

		checkCuda(cudaSetDevice(tid));
		CopyToConstantMemory(c0, c1);

		checkCuda(cudaMallocPitch((void**)&d_s_Uolds[tid], &pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2)));
		checkCuda(cudaMallocPitch((void**)&d_s_Unews[tid], &pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2)));
		checkCuda(cudaMallocPitch((void**)&d_left_receive_buffer[tid], &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));
		checkCuda(cudaMallocPitch((void**)&d_right_receive_buffer[tid], &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));
		checkCuda(cudaMallocPitch((void**)&d_left_send_buffer[tid], &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));
		checkCuda(cudaMallocPitch((void**)&d_right_send_buffer[tid], &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));
	}
}

	// Copy data from host to the device
	double HtD_timer = 0.;
	HtD_timer -= omp_get_wtime();
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		if (tid < number_of_devices) {
			checkCuda(cudaSetDevice(tid));
			checkCuda(cudaMemcpy2D(d_s_Uolds[tid], pitch_bytes, h_s_Uolds[tid], dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_Nz+2)), cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy2D(d_s_Unews[tid], pitch_bytes, h_s_Unews[tid], dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_Nz+2)), cudaMemcpyHostToDevice));
		}
	}
	HtD_timer += omp_get_wtime();

	int pitch = pitch_bytes/dt_size;
	int gc_pitch = pitch_gc_bytes/dt_size;

  // GPU kernel launch parameters
	dim3 threads_per_block(blockX, blockY, blockZ);
	unsigned int blocksInX = getBlock(Nx, blockX);
	unsigned int blocksInY = getBlock(Ny, blockY);
	unsigned int blocksInZ = getBlock(_Nz-2, k_loop);
	dim3 thread_blocks(blocksInX, blocksInY, blocksInZ);
	dim3 thread_blocks_halo(blocksInX, blocksInY);

	unsigned int ghost_width = 1;

	omp_set_dynamic(0);

	double compute_timer = 0.;
  compute_timer -= omp_get_wtime();

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		for(unsigned int iterations = 0; iterations < max_iters; iterations++) {

		#pragma omp barrier
		// Compute right boundary on devices 0-2
		if (tid < 3) {
			int kstart = (_Nz+1)-ghost_width;
			int kstop = _Nz+1;

			int i = 0;

			if (tid == 0) { i = 0; }
			if (tid == 1) { i = 1; }
			if (tid == 2) { i = 2; }

			checkCuda(cudaSetDevice(i));

			ComputeInnerPointsAsync(thread_blocks_halo, threads_per_block, 
			right_send_stream[i], d_s_Unews[i], d_s_Uolds[i], pitch, Nx, Ny, _Nz, kstart, kstop);

			CopyBoundaryRegionToGhostCellAsync(thread_blocks_halo, threads_per_block, 
			right_send_stream[i], d_s_Unews[i], d_right_send_buffer[i], Nx, Ny, _Nz, pitch, gc_pitch, 0);

			checkCuda(cudaMemcpy2DAsync(right_send_buffer[i], dt_size*(Nx+2), 
			d_right_send_buffer[i], pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH), cudaMemcpyDefault, right_send_stream[i]));
		}

		// Compute left boundary on devices 1-3
		if (tid > 0 && tid < 4) {
			int kstart = 1;
			int kstop  = 1+ghost_width;

			checkCuda(cudaSetDevice(tid));

			ComputeInnerPointsAsync(thread_blocks_halo, threads_per_block,
					left_send_stream[tid], d_s_Unews[tid], d_s_Uolds[tid], pitch, Nx, Ny, _Nz, kstart, kstop);

			CopyBoundaryRegionToGhostCellAsync(thread_blocks_halo, threads_per_block, 
					left_send_stream[tid], d_s_Unews[tid], d_left_send_buffer[tid], Nx, Ny, _Nz, pitch, gc_pitch, 1);

			checkCuda(cudaMemcpy2DAsync(left_send_buffer[tid], dt_size*(Nx+2), 
			d_left_send_buffer[tid], pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH), cudaMemcpyDefault, left_send_stream[tid]));
		}

		#pragma omp barrier

    // Compute inner nodes for device 0
    if (tid == 6) {
      int kstart = 1;
      int kstop = (_Nz+1)-ghost_width;
			
      ComputeInnerPointsAsync(thread_blocks, threads_per_block, 
      compute_stream_6, d_s_Unews[0], d_s_Uolds[0], pitch, Nx, Ny, _Nz, kstart, kstop);
    }

    // Compute inner nodes for device 1 & 2
    if (tid == 7) {
      int kstart = 1+ghost_width;
      int kstop = (_Nz+1)-ghost_width;

      ComputeInnerPointsAsync(thread_blocks, threads_per_block, 
        compute_stream_7, d_s_Unews[1], d_s_Uolds[1], pitch, Nx, Ny, _Nz, kstart, kstop);
    }

    if (tid == 8) {
      int kstart = 1+ghost_width;
      int kstop = (_Nz+1)-ghost_width;

      ComputeInnerPointsAsync(thread_blocks, threads_per_block, 
        compute_stream_8, d_s_Unews[2], d_s_Uolds[2], pitch, Nx, Ny, _Nz, kstart, kstop);
    }

    // Compute inner nodes for device 3
    if (tid == 9) {
      int kstart = 1+ghost_width;
      int kstop = _Nz+1;

      ComputeInnerPointsAsync(thread_blocks, threads_per_block, 
      compute_stream_9, d_s_Unews[3], d_s_Uolds[3], pitch, Nx, Ny, _Nz, kstart, kstop);
    }

		#pragma omp barrier

		// Copy boundary data to left side on device 1-3
		if (tid > 0 && tid < 4) {

			while (true) {
				checkCuda(cudaSetDevice(tid-1));

				if (cudaStreamQuery(right_send_stream[tid-1]) == cudaSuccess) {
					checkCuda(cudaSetDevice(tid));

					checkCuda(cudaMemcpy2DAsync(d_left_receive_buffer[tid], pitch_gc_bytes, 
						right_send_buffer[tid-1], dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_GC_DEPTH)), cudaMemcpyDefault, left_receive_stream[tid]));

					CopyGhostCellToBoundaryRegionAsync(thread_blocks_halo, threads_per_block,
						left_receive_stream[tid], d_s_Unews[tid], d_left_receive_buffer[tid], Nx, Ny, _Nz, pitch, gc_pitch, 1);

					break;
				}
			}
		}

		// Copy boundary data to right side on device 0-2
		if (tid < 3) {

			while (true) {
				checkCuda(cudaSetDevice(tid+1));
				
				if (cudaStreamQuery(left_send_stream[tid+1]) == cudaSuccess) {
					checkCuda(cudaSetDevice(tid));

					checkCuda(cudaMemcpy2DAsync(d_right_receive_buffer[tid], pitch_gc_bytes, 
				left_send_buffer[tid+1], dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_GC_DEPTH)), cudaMemcpyDefault, right_receive_stream[tid]));

					CopyGhostCellToBoundaryRegionAsync(thread_blocks_halo, threads_per_block, 
						right_receive_stream[tid], d_s_Unews[tid], d_right_receive_buffer[tid], Nx, Ny, _Nz, pitch, gc_pitch, 0);
				
				break;
				}
			}
		}

		// Swap pointers on the host
		#pragma omp barrier
		if (tid < number_of_devices) {
			checkCuda(cudaDeviceSynchronize());
			swap(REAL*, d_s_Unews[tid], d_s_Uolds[tid]);
		}
	}
}

	compute_timer += omp_get_wtime();

  // Copy data from device to host
	double DtH_timer = 0;
  DtH_timer -= omp_get_wtime();
	#pragma omp parallel for num_threads(number_of_devices)
	for (int i = 0; i < number_of_devices; i++) {
		checkCuda(cudaSetDevice(i));
		checkCuda(cudaMemcpy2D(h_s_Uolds[i], dt_size*(Nx+2), d_s_Uolds[i], pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2), cudaMemcpyDeviceToHost));
	}
	DtH_timer += omp_get_wtime();

	#pragma omp parallel for num_threads(number_of_devices)
	for (int i = 0; i < number_of_devices; i++) {
		merge_domains(h_s_Uolds[i], h_Uold, Nx, Ny, _Nz, i);
	}

   	// Calculate on host
#if defined(DEBUG) || defined(_DEBUG)
	cpu_heat3D(u_new, u_old, c0, c1, max_iters, Nx, Ny, Nz);
#endif

  float gflops = CalcGflops(compute_timer, max_iters, Nx, Ny, Nz);
  PrintSummary("3D Heat (7-pt)", "Plane sweeping", compute_timer, HtD_timer, DtH_timer, gflops, max_iters, Nx);

  REAL t = max_iters * dt;
  CalcError(h_Uold, u_old, t, h, Nx, Ny, Nz);

#if defined(DEBUG) || defined(_DEBUG)
    //exportToVTK(h_Uold, h, "heat3D.vtk", Nx, Ny, Nz);
#endif

    #pragma omp parallel for num_threads(number_of_devices)
    for (int i = 0; i < number_of_devices; i++) {
    	checkCuda(cudaSetDevice(i));

      checkCuda(cudaStreamDestroy(left_send_stream[i]));
      checkCuda(cudaStreamDestroy(right_send_stream[i]));
      checkCuda(cudaStreamDestroy(right_receive_stream[i]));
      checkCuda(cudaStreamDestroy(left_receive_stream[i]));

      checkCuda(cudaFree(d_s_Unews[i]));
      checkCuda(cudaFree(d_s_Uolds[i]));
      checkCuda(cudaFree(d_right_send_buffer[i]));
      checkCuda(cudaFree(d_right_receive_buffer[i]));
      checkCuda(cudaFree(d_left_send_buffer[i]));
      checkCuda(cudaFree(d_left_receive_buffer[i]));
    }

		#pragma omp parallel for num_threads(number_of_devices)
    for (int i = 0; i < number_of_devices; i++) {
    	checkCuda(cudaSetDevice(i));
      checkCuda(cudaFreeHost(h_s_Unews[i]));
      checkCuda(cudaFreeHost(h_s_Uolds[i]));
      checkCuda(cudaFreeHost(left_send_buffer[i]));
      checkCuda(cudaFreeHost(right_send_buffer[i]));
      checkCuda(cudaFreeHost(left_receive_buffer[i]));
      checkCuda(cudaFreeHost(right_receive_buffer[i]));
    }

		#pragma omp parallel for num_threads(number_of_devices)
    for (int i = 0; i < number_of_devices; i++) {
    	checkCuda(cudaSetDevice(i));
      checkCuda(cudaDeviceReset());
    }

    free(u_old);
    free(u_new);

	return 0;
}
