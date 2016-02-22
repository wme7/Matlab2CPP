all: hist_gpu_gmem_atomics hist_cpu hist_gpu_shmem_atomics

hist_cpu: hist_cpu.cu
	nvcc -O3  -o hist_cpu hist_cpu.cu 

hist_gpu_gmem_atomics: hist_gpu_gmem_atomics.cu
	nvcc -O3 -o hist_gpu_gmem_atomics -arch sm_11  hist_gpu_gmem_atomics.cu 


hist_gpu_shmem_atomics: hist_gpu_shmem_atomics.cu
	nvcc -O3 -o hist_gpu_shmem_atomics -arch sm_13  hist_gpu_shmem_atomics.cu 

clean:
	rm hist_gpu_gmem_atomics hist_cpu hist_gpu_shmem_atomics
