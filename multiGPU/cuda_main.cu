
#include "cuda_main.h"

// GPU Device Functions
__global__ void GPUInitOnDevice(float *p, float *u, int tid);
__global__ void GPUCalcFlux(float *p, float *Fp, float *Fm, int tid);
__global__ void GPUCalcState(float *p, float *u, float *Fp, float *Fm, int tid);

void Manage_Bounds(int phase, int tid, float *h_p, float *h_pl) {

	int N2 = N/2;
	int i;

	if (phase == -1) {
		// Update end regions of h_p (shared) from h_pl (private)
		if (tid == 0) {
			// Left end 
			h_p[0] = h_pl[3];
			h_p[1] = h_pl[4];
			h_p[2] = h_pl[5];
			// Right end
			h_p[(N2-1)*3] = h_pl[(N_GPU-2)*3];
			h_p[(N2-1)*3+1] = h_pl[(N_GPU-2)*3+1];
			h_p[(N2-1)*3+2] = h_pl[(N_GPU-2)*3+2];
		} 
		if (tid == 1) {
			// Left end
			h_p[N2*3] = h_pl[3];
			h_p[N2*3+1] = h_pl[4];
			h_p[N2*3+2] = h_pl[5];
			// Right end
			h_p[(N-1)*3] = h_pl[(N_GPU-2)*3];
			h_p[(N-1)*3+1] = h_pl[(N_GPU-2)*3+1];
			h_p[(N-1)*3+2] = h_pl[(N_GPU-2)*3+2];
		}
	}

	if (phase == 0) {
		// Perform Boundary Treatments on ends of h_pl (private)
		if (tid == 0) {
			// Reflective Condition on Left End
			h_pl[0] = h_pl[0];
			h_pl[1] = -h_pl[1];
			h_pl[2] = h_pl[2];
			// Copy conditions from other region
			h_pl[(N_GPU-1)*3] = h_p[N2*3];
			h_pl[(N_GPU-1)*3+1] = h_p[N2*3+1];
			h_pl[(N_GPU-1)*3+2] = h_p[N2*3+2];
		}
		if (tid == 1) {
			// Copy conditions from other region
			h_pl[0] = h_p[(N2-1)*3];
			h_pl[1] = h_p[(N2-1)*3+1];
			h_pl[2] = h_p[(N2-1)*3+2];
			// Reflective Condition on Right End
			h_pl[(N_GPU-1)*3] = h_pl[(N_GPU-2)*3];
			h_pl[(N_GPU-1)*3+1] = -h_pl[(N_GPU-2)*3+1];
			h_pl[(N_GPU-1)*3+2] = h_pl[(N_GPU-2)*3+2];
		}
	}

	if (phase == 1) {
		// Update all of h_p (shared) from h_pl (private)
		if (DEBUG) printf("Copying all data from local domains to whole domain (thread %d, phase %d)\n", tid, phase);
		if (tid == 0) {
			for (i = 0; i < N2; i++) {
				h_p[3*i] = h_pl[3*(i+1)];
				h_p[3*i+1] = h_pl[3*(i+1)+1];
				h_p[3*i+2] = h_pl[3*(i+1)+2];
			}
		}
		if (tid == 1) {
			for (i = 0; i < N2; i++) {
				h_p[3*(N2+i)] = h_pl[3*(i+1)];
				h_p[3*(N2+i)+1] = h_pl[3*(i+1)+1];
				h_p[3*(N2+i)+2] = h_pl[3*(i+1)+2];
			}
		}
	}
	
}

void Save_Results(float *h_p) {
	FILE *pFile;
	int i;
	pFile = fopen("results.txt","w");
	if (pFile != NULL) {
		for (i = 0; i < N; i++) {
			fprintf(pFile, "%d\t %g\t %g\t %g\n", i, h_p[i*3], h_p[i*3+1], h_p[i*3+2]);
		}
		fclose(pFile);
	} else {
		printf("Unable to save to file\n");
	}
}

void Manage_Comms(int phase, int tid, float **h_p, float **h_pl, float **d_p, float **d_u, float **d_Fp, float **d_Fm) {
	cudaError_t Error;
	size_t size;

	if (phase == 3) {
		// Send two cells of information from end regions of d_p (GPU) to h_pl (HOST)
		size = 6*sizeof(float);  // 2 cells, each containing 3 variables (rho, u, T)
		if (DEBUG) printf("===== Performing GPU-CPU Comms (phase %d, thread %d) ====\n", phase, tid);
		// Copy left end's data, then right end's data
		Error = cudaMemcpy(*h_pl, *d_p, size, cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (memcpy d_p -> h_pl) = %s\n", cudaGetErrorString(Error));
		Error = cudaMemcpy(*h_pl+3*N_GPU-6, *d_p+3*N_GPU-6, size, cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (memcpy d_p -> h_pl) = %s\n", cudaGetErrorString(Error));
	}

	if (phase == 2) {
		// Send two cells of information from end regions of h_pl (HOST) to d_p (GPU)
		size = 6*sizeof(float); // Again, 2 cells worth of information
		if (DEBUG) printf("===== Performing GPU-CPU Comms (phase %d, thread %d) ====\n", phase, tid);
		// Copy left end's data, then right end's data
		Error = cudaMemcpy(*d_p, *h_pl, size, cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (memcpy h_pl -> d_p) = %s\n", cudaGetErrorString(Error));
		Error = cudaMemcpy(*d_p+3*N_GPU-6, *h_pl+3*N_GPU-6, size, cudaMemcpyHostToDevice); if (DEBUG) printf("CUDA error (memcpy h_pl -> d_p) = %s\n", cudaGetErrorString(Error));
	}

	if (phase == 1) {
		size = 3*N_GPU*sizeof(float); // Copy data for all cells
		if (DEBUG) printf("===== Performing GPU-CPU Comms (phase %d, thread %d) ====\n", phase, tid);
		// Get our data from the GPU and send to host
		Error = cudaMemcpy(*h_pl, *d_p, size, cudaMemcpyDeviceToHost); if (DEBUG) printf("CUDA error (memcpy d_p -> h_pl) = %s\n", cudaGetErrorString(Error));
	}

}


void Manage_Memory(int phase, int tid, float **h_p, float **h_pl, float **d_p, float **d_u, float **d_Fp, float **d_Fm) {
	size_t size;  
	cudaError_t Error;
	if (phase == 0) {	
		// Allocate whole domain variables h_p on host (Master Thread)
		size = 3*N*sizeof(float);    // 3 variables per cell (rho, u, T)
		*h_p = (float*)malloc(size); 
	}
	if (phase == 1) {
		// Allocate local domain variables (h_pl) on host (All Threads)
		size = 3*N_GPU*sizeof(float);
		*h_pl = (float*)malloc(size); 
		// Each thread needs to assign its own GPU device
		Error = cudaSetDevice(tid);
		if (DEBUG) printf("CUDA error (cudaSetDevice) in thread %d = %s\n", tid, cudaGetErrorString(Error));
		// Allocate memory on device for local domain variables (All Threads)
		size = 3*N_GPU*sizeof(float);  
		Error = cudaMalloc((void**)d_p, size);
		if (DEBUG) printf("CUDA error (cudaMalloc d_p) in thread %d = %s\n", tid, cudaGetErrorString(Error));
 		Error = cudaMalloc((void**)d_u, size);
		if (DEBUG) printf("CUDA error (cudaMalloc d_u) in thread %d = %s\n", tid, cudaGetErrorString(Error));
		Error = cudaMalloc((void**)d_Fp, size);
		if (DEBUG) printf("CUDA error (cudaMalloc d_Fp) in thread %d = %s\n", tid, cudaGetErrorString(Error));
		Error = cudaMalloc((void**)d_Fm, size);
		if (DEBUG) printf("CUDA error (cudaMalloc d_Fm) in thread %d = %s\n", tid, cudaGetErrorString(Error));
		Error =	cudaDeviceSynchronize();
  		if (DEBUG) printf("CUDA error (Mem. Management Synchronize) in thread %d = %s\n", tid, cudaGetErrorString(Error));
 	}
	if (phase == 2) {
		// Free the memory for local domain variables (All Threads)
		free(*h_pl);
		Error = cudaFree(*d_p);
		if (DEBUG) printf("CUDA error (cudaFree d_p) in thread %d = %s\n", tid, cudaGetErrorString(Error));
		Error = cudaFree(*d_u);
		if (DEBUG) printf("CUDA error (cudaFree d_u) in thread %d = %s\n", tid, cudaGetErrorString(Error));
		Error = cudaFree(*d_Fp);
		if (DEBUG) printf("CUDA error (cudaFree d_Fp) in thread %d = %s\n", tid, cudaGetErrorString(Error));
		Error = cudaFree(*d_Fm);
		if (DEBUG) printf("CUDA error (cudaFree d_Fm) in thread %d = %s\n", tid, cudaGetErrorString(Error));
	}
	if (phase == 3) {
		// Free the whole domain variable (Master Thread)
		free(*h_p);
	}
}


__global__ void GPUInitOnDevice(float *p, float *u, int tid) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i < N_GPU) {
		if (tid == 0) {
			// Left half of shock tube
			p[3*i] = 10.0;		// Density
			p[3*i+1] = 0.0;		// Velocity
			p[3*i+2] = 1.0;		// Temperature
		} else if (tid == 1) {
			// Right half of shock tube
			p[3*i] = 1.0;		// Density
			p[3*i+1] = 0.0;		// Velocity
			p[3*i+2] = 1.0;		// Temperature
		}
		// Compute U
		u[3*i] = p[3*i];			// Density (Mass / unit volume)
		u[3*i+1] = p[3*i]*p[3*i+1];	// Momentum / unit volume
		u[3*i+2] = p[3*i]*(CV*p[3*i+2] + 0.5*p[3*i+1]*p[3*i+1]); // Energy / unit volume
	}
}



void Call_GPU_Init(float **d_p, float **d_u, int tid) {
	int threadsPerBlock = 64;
	int blocksPerGrid = (N_GPU + threadsPerBlock - 1) / threadsPerBlock;
	GPUInitOnDevice<<<blocksPerGrid, threadsPerBlock>>>(*d_p, *d_u, tid);
	if (DEBUG) printf("CUDA error (GPUInitOnDevice) in thread %d = %s\n", tid, cudaGetErrorString(cudaPeekAtLastError()));
}

__global__ void GPUCalcFlux(float *p, float *Fp, float *Fm, int tid) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;	// i = cell number in array
	float U[3], F[3], a;
	if (i < N_GPU) {
		// The compute function is the same for each cell
		U[0] = p[3*i];			// Density (Mass / unit volume)
		U[1] = p[3*i]*p[3*i+1];	// Momentum / unit volume
		U[2] = p[3*i]*(CV*p[3*i+2] + 0.5*p[3*i+1]*p[3*i+1]); // Energy / unit volume
	
		F[0] = p[3*i]*p[3*i+1];
		F[1] = p[3*i]*(p[3*i+1]*p[3*i+1] + R*p[3*i+2]);
		F[2] = p[3*i+1]*(U[2] + p[3*i]*R*p[3*i+2]);      

		a = sqrtf(GAMMA*R*p[3*i+2]);

		// Fluxes now - Pseudo-Rusanov Split Form
		Fp[3*i] = 0.5*F[0] + a*U[0];
		Fp[3*i+1] = 0.5*F[1] + a*U[1];
		Fp[3*i+2] = 0.5*F[2] + a*U[2];
		
		Fm[3*i] = 0.5*F[0] - a*U[0];
		Fm[3*i+1] = 0.5*F[1] - a*U[1];
		Fm[3*i+2] = 0.5*F[2] - a*U[2];
	}
}

void Call_GPU_Calc_Flux(float **d_p, float **d_Fp, float **d_Fm, int tid) {
	int threadsPerBlock = 64;
	int blocksPerGrid = (N_GPU + threadsPerBlock - 1) / threadsPerBlock;
	GPUCalcFlux<<<blocksPerGrid, threadsPerBlock>>>(*d_p, *d_Fp, *d_Fm, tid);
	if (DEBUG) printf("CUDA error (GPUCalcFlux) in thread %d = %s\n", tid, cudaGetErrorString(cudaPeekAtLastError()));

}

__global__ void GPUCalcState(float *p, float *u, float *Fp, float *Fm, int tid) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;	// i = cell number in array
	float FL[3], FR[3];
	if ((i>0) && (i < (N_GPU-1))) {
		// Compute net fluxes on left and right
		// Net fluxes on left
		FL[0] = Fp[3*(i-1)] + Fm[3*i];
		FL[1] = Fp[3*(i-1)+1] + Fm[3*i+1];
		FL[2] = Fp[3*(i-1)+2] + Fm[3*i+2];

		// Net fluxes on right
		FR[0] = Fp[3*i] + Fm[3*(i+1)];
		FR[1] = Fp[3*i+1] + Fm[3*(i+1)+1];
		FR[2] = Fp[3*i+2] + Fm[3*(i+1)+2];

		// Update the state in this cell
		u[3*i] = u[3*i] - (DT/DX)*(FR[0] - FL[0]);
		u[3*i+1] = u[3*i+1] - (DT/DX)*(FR[1] - FL[1]);
		u[3*i+2] = u[3*i+2] - (DT/DX)*(FR[2] - FL[2]);

		// Update p as well
		p[3*i] = u[3*i];	
		p[3*i+1] = u[3*i+1]/u[3*i];
		p[3*i+2] = ((u[3*i+2]/u[3*i])-0.5*p[3*i+1]*p[3*i+1])/CV;
	}
}

void Call_GPU_Calc_State(float **d_p, float **d_u, float **d_Fp, float **d_Fm, int tid) {
	int threadsPerBlock = 64;
	int blocksPerGrid = (N_GPU + threadsPerBlock - 1) / threadsPerBlock;
	cudaError_t Error;
	GPUCalcState<<<blocksPerGrid, threadsPerBlock>>>(*d_p, *d_u, *d_Fp, *d_Fm, tid);
	if (DEBUG) printf("CUDA error (GPUCalcState) in thread %d = %s\n", tid, cudaGetErrorString(cudaPeekAtLastError()));
	Error =	cudaDeviceSynchronize();
	if (DEBUG) printf("CUDA error (GPUCalcState Synchronize) in thread %d = %s\n", tid, cudaGetErrorString(Error));
}



