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

# Commands
all: CPU_CORE MPI_CORE GPU_CORE
	mpic++ $(CUDALDFLAGS) main.o tools2d.o heat2d.o -o heat2d.run

CPU_CORE: main.cpp
	mpic++ $(MPI_INCLUDES) $(CUDALDFLAGS) $(CFLAGS) -c main.cpp -o main.o

MPI_CORE: heat2d.cpp
	mpic++ $(MPI_INCLUDES) $(CUDALDFLAGS) $(CFLAGS) -c heat2d.cpp -o tools2d.o

GPU_CORE: heat2d.cu
	nvcc $(MPI_INCLUDES) $(NVCCFLAGS) -c heat2d.cu -o heat2d.o


clean:
	rm *.o
	rm *.run
	rm *.txt
