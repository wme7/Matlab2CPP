# Paths
MPI_INSTALL_PATH := /home/manuel/openMPI
CUDA_INSTALL_PATH := /usr/local/cuda-7.5

# Libraries
MPI_INCLUDES := -I $(MPI_INSTALL_PATH)/include
CUDA_INCLUDES := -I ${CUDA_INSTALL_PATH}/include

# gencode flags
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM37    := -gencode arch=compute_37,code=sm_37
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_FLAGS   := $(GENCODE_SM35) $(GENCODE_SM37) $(GENCODE_SM50) 

# flags
CFLAGS= -O3 -march=native #-Wall
NVCCFLAGS= -O3 $(GENCODE_FLAGS) -Xcompiler -march=native
CUDALDFLAGS= -L ${CUDA_INSTALL_PATH}/lib64 -lcuda -lcudart

# Commands
all: MPI_CORE GPU_CORE
	mpic++ $(CUDALDFLAGS) main.o laplace3d.o -o heat3d.run

MPI_CORE: main.cpp
	mpic++ $(MPI_INCLUDES) $(CUDALDFLAGS) $(CFLAGS) -c main.cpp -o main.o

GPU_CORE: heat3d.cu
	nvcc $(MPI_INCLUDES) $(NVCCFLAGS) -c heat3d.cu -o laplace3d.o

clean:
	rm *.o
	rm *.run
	rm *.txt

