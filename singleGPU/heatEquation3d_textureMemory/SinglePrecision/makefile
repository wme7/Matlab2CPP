CUDA_INSTALL_PATH := /usr/local/cuda-7.5
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart

all: MAIN

MAIN: CPU_CORE GPU_CORE
	nvcc main.o heat3d.o $(CUDA_LIB) -O3 -o heat3d.run

CPU_CORE: main.cpp
	nvcc -O3 -c main.cpp 

GPU_CORE: heat3d.cu
	nvcc -O3 -arch compute_35 -c heat3d.cu

clean:
	rm *.o
	rm *.txt
	rm *.run
