CUDA_INSTALL_PATH := /usr/local/cuda-7.5
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart

all: MAIN

MAIN: CPU_CORE GPU_CORE
	nvcc main.o heat2d.o $(CUDA_LIB) -O3 -o heat2d.run

CPU_CORE: main.cpp
	nvcc -O3 -c main.cpp 

GPU_CORE: heat2d.cu
	nvcc -O3 -arch compute_35 -c heat2d.cu

clean:
	rm *.o
	rm *.txt
	rm *.run
