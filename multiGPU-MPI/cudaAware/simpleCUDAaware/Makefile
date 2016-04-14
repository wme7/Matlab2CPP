# Paths
MPI_INSTALL_PATH := /home/manuel/openMPI
CUDA_INSTALL_PATH := /usr/local/cuda-7.5

# Libraries
MPI_INCLUDES := -I $(MPI_INSTALL_PATH)/include
CUDA_INCLUDES := -I ${CUDA_INSTALL_PATH}/include

# gencode flags
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM37    := -gencode arch=compute_37,code=sm_37
GENCODE_FLAGS   := $(GENCODE_SM35) $(GENCODE_SM37)

# flags
CFLAGS= -O3 -march=native #-Wall
NVCCFLAGS= -O3 $(GENCODE_FLAGS) -Xcompiler -march=native
CUDALDFLAGS= -L ${CUDA_INSTALL_PATH}/lib64 -lcuda -lcudart

# procedure
all: MAIN

MAIN: GPU_CORE
	mpic++ $(MPI_INCLUDES) $(CUDALDFLAGS) gpu_mpi.o -o testCUDAaware.run

GPU_CORE: check.cu
	nvcc $(MPI_INCLUDES) $(NVCCFLAGS) -c check.cu -o gpu_mpi.o

clean:
	rm *.o
	rm *.run
