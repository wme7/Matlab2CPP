#include "heat3d.h"

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

/*********************************************/
/* A method for checking error in CUDA calls */
/*********************************************/
inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
	#if defined(DEBUG) || defined(_DEBUG)
	if (error != cudaSuccess)
	{
		printf("checkCuda error at %s:%i: %s\n", file, line, cudaGetErrorString(cudaGetLastError()));
		exit(-1);
	}
	#endif

	return;
}

/**********************/
/* Main program entry */
/**********************/
int main(int argc, char** argv)
{
	REAL C;
  	unsigned int L, W, H, Nx, Ny, Nz, max_iters, blockX, blockY, blockZ;
	int rank, numberOfProcesses;

	if (argc == 12)
	{
		C = atof(argv[1]); // conductivity, here it is assumed: Cx = Cy = Cz = C.
		L = atoi(argv[2]); // domain lenght
		W = atoi(argv[3]); // domain width 
		H = atoi(argv[4]); // domain height
		Nx = atoi(argv[5]); // number cells in x-direction
		Ny = atoi(argv[6]); // number cells in x-direction
		Nz = atoi(argv[7]); // number cells in x-direction
		max_iters = atoi(argv[8]); // number of iterations / time steps
		blockX = atoi(argv[9]); // block size in the i-direction
		blockY = atoi(argv[10]); // block size in the j-direction
		blockZ = atoi(argv[11]); // block size in the k-direction
	}
	else
	{
		printf("Usage: %s diffCoef L W H nx ny nz i block_x block_y block_z\n", argv[0]);
		exit(1);
	}

	InitializeMPI(&argc, &argv, &rank, &numberOfProcesses);
	AssignDevices(rank);
	ECCCheck(rank);

	unsigned int R;	REAL dx, dy, dz, dt, kx, ky, kz, tFinal;

	dx = (REAL)L/(Nx+1); // dx, cell size
	dy = (REAL)W/(Ny+1); // dy, cell size
	dz = (REAL)H/(Nz+1); // dz, cell size
	dt = 1/(2*C*(1/dx/dx+1/dy/dy+1/dz/dz)); // dt, fix time step size
	kx = C*dt/(dx*dx); // numerical conductivity
	ky = C*dt/(dy*dy); // numerical conductivity
	kz = C*dt/(dz*dz); // numerical conductivity
	tFinal = dt*max_iters; //printf("Final time: %g\n",tFinal);
	R = 1; // halo regions width (in cells size)

	// Copy constants to Constant Memory on the GPUs
	CopyToConstantMemory(kx, ky, kz);

	// Decompose along the z-axis
	const int _Nz = Nz/numberOfProcesses;
  	const int dt_size = sizeof(REAL);

	// Host memory allocations
	REAL *h_u, *h_un;

	// Allocate global domains in host
	h_u  = (REAL*)malloc(dt_size*(Nx+2*R)*(Ny+2*R)*(Nz+2*R));
	h_un = (REAL*)malloc(dt_size*(Nx+2*R)*(Ny+2*R)*(Nz+2*R));

	// Set Initial condition in global domain
	init(h_u , dx, dy, dz, Nx+2*R, Ny+2*R, Nz+2*R);

	// Allocate and generate host subdomains
	REAL *h_s_u, *h_s_un, *h_s_recvbuff[numberOfProcesses];
	REAL *l_send_buffer, *l_recv_buffer;
	REAL *r_send_buffer, *r_recv_buffer;

	h_s_u  = (REAL*)malloc(dt_size*(Nx+2*R)*(Ny+2*R)*(_Nz+2*R));
	h_s_un = (REAL*)malloc(dt_size*(Nx+2*R)*(Ny+2*R)*(_Nz+2*R));

    checkCuda(cudaHostAlloc((void**)&h_s_un, dt_size*(Nx+2*R)*(Ny+2*R)*(_Nz+2*R), cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&h_s_u , dt_size*(Nx+2*R)*(Ny+2*R)*(_Nz+2*R), cudaHostAllocPortable));

	if (rank == 0) {
    	for (int i = 0; i < numberOfProcesses; i++) {
        	h_s_recvbuff[i] = (REAL*)malloc(dt_size*(Nx+2)*(Ny+2)*(_Nz+2));
        	checkCuda(cudaHostAlloc((void**)&h_s_recvbuff[i], dt_size*(Nx+2)*(Ny+2)*(_Nz+2), cudaHostAllocPortable));
    	}
  	}

  	// Allocate exchange hallo subdomains
    r_send_buffer = (REAL*)malloc(dt_size*(Nx+2*R)*(Ny+2*R)*R);
    l_send_buffer = (REAL*)malloc(dt_size*(Nx+2*R)*(Ny+2*R)*R);
    r_recv_buffer = (REAL*)malloc(dt_size*(Nx+2*R)*(Ny+2*R)*R);
    l_recv_buffer = (REAL*)malloc(dt_size*(Nx+2*R)*(Ny+2*R)*R);

    checkCuda(cudaHostAlloc((void**)&r_send_buffer, dt_size*(Nx+2*R)*(Ny+2*R)*R, cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&l_send_buffer, dt_size*(Nx+2*R)*(Ny+2*R)*R, cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&r_recv_buffer, dt_size*(Nx+2*R)*(Ny+2*R)*R, cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&l_recv_buffer, dt_size*(Nx+2*R)*(Ny+2*R)*R, cudaHostAllocPortable));

    init_subdomain(h_s_u, h_u, Nx, Ny, _Nz, rank);

    if (DEBUG) printf("check point %d in rank %d\n",1,rank);

	// GPU stream operations
	cudaStream_t compute_stream;
	cudaStream_t r_send_stream, r_recv_stream;
	cudaStream_t l_send_stream, l_recv_stream;

	checkCuda(cudaStreamCreate(&compute_stream));
	checkCuda(cudaStreamCreate(&r_send_stream));
	checkCuda(cudaStreamCreate(&l_send_stream));
	checkCuda(cudaStreamCreate(&r_recv_stream));
	checkCuda(cudaStreamCreate(&l_recv_stream));

	// GPU Memory Operations
	size_t pitch_bytes, pitch_gc_bytes;

    REAL *d_s_u, *d_s_un;
    REAL *d_r_send_buffer, *d_l_send_buffer;
    REAL *d_r_recv_buffer, *d_l_recv_buffer;

    checkCuda(cudaMallocPitch((void**)&d_s_u , &pitch_bytes, dt_size*(Nx+2*R), (Ny+2*R)*(_Nz+2*R)));
    checkCuda(cudaMallocPitch((void**)&d_s_un, &pitch_bytes, dt_size*(Nx+2*R), (Ny+2*R)*(_Nz+2*R)));

    checkCuda(cudaMallocPitch((void**)&d_l_send_buffer, &pitch_gc_bytes, dt_size*(Nx+2*R), (Ny+2*R)*R));
    checkCuda(cudaMallocPitch((void**)&d_l_recv_buffer, &pitch_gc_bytes, dt_size*(Nx+2*R), (Ny+2*R)*R));
    checkCuda(cudaMallocPitch((void**)&d_r_send_buffer, &pitch_gc_bytes, dt_size*(Nx+2*R), (Ny+2*R)*R));
    checkCuda(cudaMallocPitch((void**)&d_r_recv_buffer, &pitch_gc_bytes, dt_size*(Nx+2*R), (Ny+2*R)*R));

	// Copy subdomains from host to device and get walltime
	double HtD_timer = 0.;

	CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
	HtD_timer -= MPI_Wtime();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    checkCuda(cudaMemcpy2D(d_s_u , pitch_bytes, h_s_u , dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_Nz+2)), cudaMemcpyDefault));
    checkCuda(cudaMemcpy2D(d_s_un, pitch_bytes, h_s_un, dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_Nz+2)), cudaMemcpyDefault));

    if (DEBUG) printf("check point %d in rank %d\n",2,rank);

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
	HtD_timer += MPI_Wtime();
	CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

	unsigned int ghost_width = 1;

	int pitch = pitch_bytes/dt_size;
	int gc_pitch = pitch_gc_bytes/dt_size;

    // GPU kernel launch parameters
	dim3 threads_per_block(blockX, blockY, blockZ);
	unsigned int blocksInX = getBlock(Nx, blockX);
	unsigned int blocksInY = getBlock(Ny, blockY);
	unsigned int blocksInZ = getBlock(_Nz-2, k_loop);

	dim3 thread_blocks(blocksInX, blocksInY, blocksInZ);
	dim3 thread_blocks_halo(blocksInX, blocksInY);

	//MPI_Status status;
	MPI_Status status[numberOfProcesses];
	MPI_Request gather_send_request[numberOfProcesses];
	MPI_Request r_send_request[numberOfProcesses], l_send_request[numberOfProcesses];
	MPI_Request r_recv_request[numberOfProcesses], l_recv_request[numberOfProcesses];

	double compute_timer = 0.;

	CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    compute_timer -= MPI_Wtime();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

	for(unsigned int iterations = 1; iterations < max_iters; iterations++)
	{
		// Compute right boundary on devices 0-2, send to devices 1-n-1
		if (rank < numberOfProcesses-1)
		{
			int kstart = (_Nz+1)-ghost_width;
			int kstop = _Nz+1;

			ComputeInnerPointsAsync(thread_blocks_halo, threads_per_block, r_send_stream, d_s_un, d_s_u, pitch, Nx, Ny, _Nz, kstart, kstop);
			CopyBoundaryRegionToGhostCellAsync(thread_blocks_halo, threads_per_block, r_send_stream, d_s_un, d_r_send_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 0);

			checkCuda(cudaMemcpy2DAsync(r_send_buffer, dt_size*(Nx+2), d_r_send_buffer, pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(R), cudaMemcpyDefault, r_send_stream));
			checkCuda(cudaStreamSynchronize(r_send_stream));

			CHECK_MPI(MPI_Isend(r_send_buffer, (Nx+2)*(Ny+2)*(R), MPI_CUSTOM_REAL, rank+1, 0, MPI_COMM_WORLD, &r_send_request[rank]));
		}

		if (rank > 0)
		{
			int kstart = 1;
			int kstop  = 1+ghost_width;

			ComputeInnerPointsAsync(thread_blocks_halo, threads_per_block, l_send_stream, d_s_un, d_s_u, pitch, Nx, Ny, _Nz, kstart, kstop);
			CopyBoundaryRegionToGhostCellAsync(thread_blocks_halo, threads_per_block, l_send_stream, d_s_un, d_l_send_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 1);

			checkCuda(cudaMemcpy2DAsync(l_send_buffer, dt_size*(Nx+2), d_l_send_buffer, pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(R), cudaMemcpyDefault, l_send_stream));
			checkCuda(cudaStreamSynchronize(l_send_stream));

			CHECK_MPI(MPI_Isend(l_send_buffer, (Nx+2)*(Ny+2)*(R), MPI_CUSTOM_REAL, rank-1, 1, MPI_COMM_WORLD, &l_send_request[rank]));
		}

		// Compute inner points for device 0
		if (rank == 0)
		{
			int kstart = 1;
			int kstop = (_Nz+1)-ghost_width;

			ComputeInnerPointsAsync(thread_blocks, threads_per_block, compute_stream, d_s_un, d_s_u, pitch, Nx, Ny, _Nz, kstart, kstop);
		}

		// Compute inner points for device 1 and n-1
		if (rank > 0 && rank < numberOfProcesses-1)
		{
			int kstart = 1+ghost_width;
			int kstop = (_Nz+1)-ghost_width;

			ComputeInnerPointsAsync(thread_blocks, threads_per_block, compute_stream, d_s_un, d_s_u, pitch, Nx, Ny, _Nz, kstart, kstop);
		}

		// Compute inner points for device n
		if (rank == numberOfProcesses-1)
		{
			int kstart = 1+ghost_width;
			int kstop = _Nz+1;

			ComputeInnerPointsAsync(thread_blocks, threads_per_block, compute_stream, d_s_un, d_s_u, pitch, Nx, Ny, _Nz, kstart, kstop);
		}

		// Receive data from 0-2
		if (rank < numberOfProcesses-1)
		{
			CHECK_MPI(MPI_Irecv(r_recv_buffer, (Nx+2)*(Ny+2)*(R), MPI_CUSTOM_REAL, rank+1, 1, MPI_COMM_WORLD, &r_recv_request[rank]));
		}

		// Receive data from 1-3
		if (rank > 0)
		{
			CHECK_MPI(MPI_Irecv(l_recv_buffer, (Nx+2)*(Ny+2)*(R), MPI_CUSTOM_REAL, rank-1, 0, MPI_COMM_WORLD, &l_recv_request[rank]));
		}

		// Receive data from 0-2
		if (rank < numberOfProcesses-1)
		{
			CHECK_MPI(MPI_Wait(&r_recv_request[rank], MPI_STATUS_IGNORE));

			checkCuda(cudaMemcpy2DAsync(d_r_recv_buffer, pitch_gc_bytes, r_recv_buffer, dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(R)), cudaMemcpyDefault, r_recv_stream));
			CopyGhostCellToBoundaryRegionAsync(thread_blocks_halo, threads_per_block, r_recv_stream, d_s_un, d_r_recv_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 0);
		}

		// Receive data from 1-3
		if (rank > 0)
		{
			CHECK_MPI(MPI_Wait(&l_recv_request[rank], MPI_STATUS_IGNORE));

			checkCuda(cudaMemcpy2DAsync(d_l_recv_buffer, pitch_gc_bytes, l_recv_buffer, dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(R)), cudaMemcpyDefault, l_recv_stream));
			CopyGhostCellToBoundaryRegionAsync(thread_blocks_halo, threads_per_block, l_recv_stream, d_s_un, d_l_recv_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 1);
		}

		if (rank < numberOfProcesses-1)
		{
			CHECK_MPI(MPI_Wait(&r_send_request[rank], MPI_STATUS_IGNORE));
		}

		if (rank > 0)
		{
			CHECK_MPI(MPI_Wait(&l_send_request[rank], MPI_STATUS_IGNORE));
		}

		// Swap pointers on the host
		checkCuda(cudaDeviceSynchronize());
		SWAP(REAL*, d_s_un, d_s_u);
	}

	CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
	compute_timer += MPI_Wtime();
	CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

	// Copy data from device to host
	double DtH_timer = 0;

	CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    DtH_timer -= MPI_Wtime();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

	checkCuda(cudaMemcpy2D(h_s_u, dt_size*(Nx+2), d_s_u, pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2), cudaMemcpyDefault));

	CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
	DtH_timer += MPI_Wtime();
	CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    // Gather results from subdomains
    CHECK_MPI(MPI_Isend(h_s_u, (Nx+2)*(Ny+2)*(_Nz+2), MPI_CUSTOM_REAL, 0, 0, MPI_COMM_WORLD, &gather_send_request[rank]));

    if (DEBUG) printf("check point %d in rank %d\n",3,rank);

	if (rank == 0)
	{
		for (int i = 0; i < numberOfProcesses; i++)
		{
			CHECK_MPI(MPI_Recv(h_s_recvbuff[i], (Nx+2)*(Ny+2)*(_Nz+2), MPI_CUSTOM_REAL, i, 0, MPI_COMM_WORLD, &status[rank]));
			merge_subdomain(h_s_recvbuff[i], h_un, Nx, Ny, _Nz, i);
		}
		if (WRITE) Save3D(h_un, Nx+2*R, Ny+2*R, Nz+2*R);
	}

	if (rank == 0)
	{
		float gflops = CalcGflops(compute_timer, max_iters, Nx, Ny, Nz);
		PrintSummary("HeatEq3D (7-pt)", "Plane sweeping", compute_timer, tFinal, HtD_timer, DtH_timer, gflops, max_iters, Nx+2, Ny+2, Nz+2);
		CalcError(h_un, tFinal, dx, dy, dz, Nx, Ny, Nz);
	}

	if (DEBUG) printf("check point %d in rank %d\n",4,rank);

	// Finalize MPI
	Finalize();

  // Free device memory
    checkCuda(cudaFree(d_s_un));
    checkCuda(cudaFree(d_s_u));
    checkCuda(cudaFree(d_r_send_buffer));
    checkCuda(cudaFree(d_l_send_buffer));
    checkCuda(cudaFree(d_r_recv_buffer));
    checkCuda(cudaFree(d_l_recv_buffer));

    // Free host memory
    checkCuda(cudaFreeHost(h_s_un));
    checkCuda(cudaFreeHost(h_s_u));

	if (rank == 0) {
		for (int i = 0; i < numberOfProcesses; i++) checkCuda(cudaFreeHost(h_s_recvbuff[i])); 
	}

    checkCuda(cudaFreeHost(l_send_buffer));
    checkCuda(cudaFreeHost(l_recv_buffer));
    checkCuda(cudaFreeHost(r_send_buffer));
    checkCuda(cudaFreeHost(r_recv_buffer));

    checkCuda(cudaDeviceReset());

    free(h_un);
    free(h_u);

	return 0;
}
