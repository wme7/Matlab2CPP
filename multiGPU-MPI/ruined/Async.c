#include "heat3d.h"

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

////////////////////////////////////////////////////////////////////////////////
// A method for checking error in CUDA calls
////////////////////////////////////////////////////////////////////////////////
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

///////////////////////
// Main program entry
///////////////////////
int main(int argc, char** argv)
{
	unsigned int max_iters, Nx, Ny, Nz, blockX, blockY, blockZ;
	int rank, numberOfProcesses;

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

	InitializeMPI(&argc, &argv, &rank, &numberOfProcesses);
	AssignDevices(rank);
	ECCCheck(rank);

	// Define constants
	const REAL L = 1.0;
	const REAL dx = L/(Nx+1);
	const REAL dt = dx*dx/6.0;
	const REAL beta = dt/(dx*dx);
	const REAL kx = beta;
	const REAL ky = beta;
	const REAL kz = beta;

	// Copy constants to Constant Memory on the GPUs
	CopyToConstantMemory(kx, ky, kz);

	// Decompose along the z-axis
	const int _Nz = Nz/numberOfProcesses;
  	const int dt_size = sizeof(REAL);

	// Host memory allocations
	REAL *h_un, *h_u;
	REAL *h_Uold;

	h_un = (REAL*)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(Nz+2));
	h_u  = (REAL*)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(Nz+2));

	if (rank == 0)
	{
	h_Uold = (REAL*)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(Nz+2)); 
	}

	init(h_u, h_un, dx, Nx, Ny, Nz);

	// Allocate and generate host subdomains (building pinned accesible by all CUDA contexts)
	REAL *h_s_Uolds, *h_s_Unews, *h_s_recvbuff[numberOfProcesses];
	REAL *l_send_buffer, *l_recv_buffer;
	REAL *r_send_buffer, *r_recv_buffer;

	h_s_Unews = (REAL*)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(_Nz+2));
	h_s_Uolds = (REAL*)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(_Nz+2));
	checkCuda(cudaHostAlloc((void**)&h_s_Unews, dt_size*(Nx+2)*(Ny+2)*(_Nz+2), cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&h_s_Uolds, dt_size*(Nx+2)*(Ny+2)*(_Nz+2), cudaHostAllocPortable));

	if (rank == 0)
	{
		for (int i = 0; i < numberOfProcesses; i++)
		{
		    h_s_recvbuff[i] = (REAL*)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(_Nz+2));
		    checkCuda(cudaHostAlloc((void**)&h_s_recvbuff[i], dt_size*(Nx+2)*(Ny+2)*(_Nz+2), cudaHostAllocPortable));
		}
	}

    r_send_buffer = (REAL*)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(RADIUS));
    l_send_buffer = (REAL*)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(RADIUS));
    r_recv_buffer = (REAL*)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(RADIUS));
    l_recv_buffer = (REAL*)malloc(sizeof(REAL)*(Nx+2)*(Ny+2)*(RADIUS));
    checkCuda(cudaHostAlloc((void**)&r_send_buffer, dt_size*(Nx+2)*(Ny+2)*(RADIUS), cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&l_send_buffer, dt_size*(Nx+2)*(Ny+2)*(RADIUS), cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&r_recv_buffer, dt_size*(Nx+2)*(Ny+2)*(RADIUS), cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&l_recv_buffer, dt_size*(Nx+2)*(Ny+2)*(RADIUS), cudaHostAllocPortable));

    init_subdomain(h_s_Uolds, h_u, Nx, Ny, _Nz, rank);

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

    REAL *d_s_Unews, *d_s_Uolds;
    REAL *d_r_send_buffer, *d_l_send_buffer;
    REAL *d_r_recv_buffer, *d_l_recv_buffer;

    checkCuda(cudaMallocPitch((void**)&d_s_Uolds, &pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2)));
    checkCuda(cudaMallocPitch((void**)&d_s_Unews, &pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2)));
    checkCuda(cudaMallocPitch((void**)&d_l_send_buffer, &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(RADIUS)));
    checkCuda(cudaMallocPitch((void**)&d_l_recv_buffer, &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(RADIUS)));
    checkCuda(cudaMallocPitch((void**)&d_r_send_buffer, &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(RADIUS)));
    checkCuda(cudaMallocPitch((void**)&d_r_recv_buffer, &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(RADIUS)));

	// Copy subdomains from host to device and get walltime
	double HtD_timer = 0.;

	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
	HtD_timer -= MPI_Wtime();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    checkCuda(cudaMemcpy2D(d_s_Uolds, pitch_bytes, h_s_Uolds, dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_Nz+2)), cudaMemcpyDefault));
    checkCuda(cudaMemcpy2D(d_s_Unews, pitch_bytes, h_s_Unews, dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_Nz+2)), cudaMemcpyDefault));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
	HtD_timer += MPI_Wtime();
	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

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

	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    compute_timer -= MPI_Wtime();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

	for(unsigned int iterations = 0; iterations < max_iters; iterations++)
	{
		// Compute right boundary on devices 0-2, send to devices 1-n-1
		if (rank < numberOfProcesses-1)
		{
			int kstart = (_Nz+1)-ghost_width;
			int kstop = _Nz+1;

			ComputeInnerPointsAsync(thread_blocks_halo, threads_per_block, r_send_stream, d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, kstart, kstop);
			CopyBoundaryRegionToGhostCellAsync(thread_blocks_halo, threads_per_block, r_send_stream, d_s_Unews, d_r_send_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 0);

			checkCuda(cudaMemcpy2DAsync(r_send_buffer, dt_size*(Nx+2), d_r_send_buffer, pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(RADIUS), cudaMemcpyDefault, r_send_stream));
			checkCuda(cudaStreamSynchronize(r_send_stream));

			MPI_CHECK(MPI_Isend(r_send_buffer, (Nx+2)*(Ny+2)*(RADIUS), MPI_CUSTOM_REAL, rank+1, 0, MPI_COMM_WORLD, &r_send_request[rank]));
		}

		if (rank > 0)
		{
			int kstart = 1;
			int kstop  = 1+ghost_width;

			ComputeInnerPointsAsync(thread_blocks_halo, threads_per_block, l_send_stream, d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, kstart, kstop);
			CopyBoundaryRegionToGhostCellAsync(thread_blocks_halo, threads_per_block, l_send_stream, d_s_Unews, d_l_send_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 1);

			checkCuda(cudaMemcpy2DAsync(l_send_buffer, dt_size*(Nx+2), d_l_send_buffer, pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(RADIUS), cudaMemcpyDefault, l_send_stream));
			checkCuda(cudaStreamSynchronize(l_send_stream));

			MPI_CHECK(MPI_Isend(l_send_buffer, (Nx+2)*(Ny+2)*(RADIUS), MPI_CUSTOM_REAL, rank-1, 1, MPI_COMM_WORLD, &l_send_request[rank]));
		}

		// Compute inner points for device 0
		if (rank == 0)
		{
			int kstart = 1;
			int kstop = (_Nz+1)-ghost_width;

			ComputeInnerPointsAsync(thread_blocks, threads_per_block, compute_stream, d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, kstart, kstop);
		}

		// Compute inner points for device 1 and n-1
		if (rank > 0 && rank < numberOfProcesses-1)
		{
			int kstart = 1+ghost_width;
			int kstop = (_Nz+1)-ghost_width;

			ComputeInnerPointsAsync(thread_blocks, threads_per_block, compute_stream, d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, kstart, kstop);
		}

		// Compute inner points for device n
		if (rank == numberOfProcesses-1)
		{
			int kstart = 1+ghost_width;
			int kstop = _Nz+1;

			ComputeInnerPointsAsync(thread_blocks, threads_per_block, compute_stream, d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, kstart, kstop);
		}

		// Receive data from 0-2
		if (rank < numberOfProcesses-1)
		{
			MPI_CHECK(MPI_Irecv(r_recv_buffer, (Nx+2)*(Ny+2)*(RADIUS), MPI_CUSTOM_REAL, rank+1, 1, MPI_COMM_WORLD, &r_recv_request[rank]));
		}

		// Receive data from 1-3
		if (rank > 0)
		{
			MPI_CHECK(MPI_Irecv(l_recv_buffer, (Nx+2)*(Ny+2)*(RADIUS), MPI_CUSTOM_REAL, rank-1, 0, MPI_COMM_WORLD, &l_recv_request[rank]));
		}

		// Receive data from 0-2
		if (rank < numberOfProcesses-1)
		{
			MPI_CHECK(MPI_Wait(&r_recv_request[rank], MPI_STATUS_IGNORE));

			checkCuda(cudaMemcpy2DAsync(d_r_recv_buffer, pitch_gc_bytes, r_recv_buffer, dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(RADIUS)), cudaMemcpyDefault, r_recv_stream));
			CopyGhostCellToBoundaryRegionAsync(thread_blocks_halo, threads_per_block, r_recv_stream, d_s_Unews, d_r_recv_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 0);
		}

		// Receive data from 1-3
		if (rank > 0)
		{
			MPI_CHECK(MPI_Wait(&l_recv_request[rank], MPI_STATUS_IGNORE));

			checkCuda(cudaMemcpy2DAsync(d_l_recv_buffer, pitch_gc_bytes, l_recv_buffer, dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(RADIUS)), cudaMemcpyDefault, l_recv_stream));
			CopyGhostCellToBoundaryRegionAsync(thread_blocks_halo, threads_per_block, l_recv_stream, d_s_Unews, d_l_recv_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 1);
		}

		if (rank < numberOfProcesses-1)
		{
			MPI_CHECK(MPI_Wait(&r_send_request[rank], MPI_STATUS_IGNORE));
		}

		if (rank > 0)
		{
			MPI_CHECK(MPI_Wait(&l_send_request[rank], MPI_STATUS_IGNORE));
		}

		// Swap pointers on the host
		checkCuda(cudaDeviceSynchronize());
		swap(REAL*, d_s_Unews, d_s_Uolds);
	}

	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
	compute_timer += MPI_Wtime();
	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

	// Copy data from device to host
	double DtH_timer = 0;

	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    DtH_timer -= MPI_Wtime();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

	checkCuda(cudaMemcpy2D(h_s_Uolds, dt_size*(Nx+2), d_s_Uolds, pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2), cudaMemcpyDefault));

	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
	DtH_timer += MPI_Wtime();
	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Gather results from subdomains
    MPI_CHECK(MPI_Isend(h_s_Uolds, (Nx+2)*(Ny+2)*(_Nz+2), MPI_CUSTOM_REAL, 0, 0, MPI_COMM_WORLD, &gather_send_request[rank]));

	if (rank == 0)
	{
		for (int i = 0; i < numberOfProcesses; i++)
		{
			MPI_CHECK(MPI_Recv(h_s_recvbuff[i], (Nx+2)*(Ny+2)*(_Nz+2), MPI_CUSTOM_REAL, i, 0, MPI_COMM_WORLD, &status[rank]));
			merge_domains(h_s_recvbuff[i], h_Uold, Nx, Ny, _Nz, i);
		}
		// print solution to file
		Save3D(h_Uold, Nx+2, Ny+2, Nz+2);
	}

	// Calculate on host
	if (rank == 0)
	{
		cpu_heat3D(h_un, h_u, kx, ky, kz, max_iters, Nx, Ny, Nz);
	}

	if (rank == 0)
	{
		float gflops = CalcGflops(compute_timer, max_iters, Nx, Ny, Nz);
		PrintSummary("3D Heat (7-pt)", "Plane sweeping", compute_timer, HtD_timer, DtH_timer, gflops, max_iters, Nx);

		REAL t = max_iters * dt;
		CalcError(h_Uold, h_u, t, dx, Nx, Ny, Nz);
	}

	Finalize();

  // Free device memory
    checkCuda(cudaFree(d_s_Unews));
    checkCuda(cudaFree(d_s_Uolds));
    checkCuda(cudaFree(d_r_send_buffer));
    checkCuda(cudaFree(d_l_send_buffer));
    checkCuda(cudaFree(d_r_recv_buffer));
    checkCuda(cudaFree(d_l_recv_buffer));

    // Free host memory
    checkCuda(cudaFreeHost(h_s_Unews));
    checkCuda(cudaFreeHost(h_s_Uolds));


	if (rank == 0)
	{
		for (int i = 0; i < numberOfProcesses; i++)
		{
			checkCuda(cudaFreeHost(h_s_recvbuff[i]));
		}

	free(h_Uold);
	}

    checkCuda(cudaFreeHost(l_send_buffer));
    checkCuda(cudaFreeHost(l_recv_buffer));
    checkCuda(cudaFreeHost(r_send_buffer));
    checkCuda(cudaFreeHost(r_recv_buffer));

    checkCuda(cudaDeviceReset());

    free(h_u);
    free(h_un);

	return 0;
}
