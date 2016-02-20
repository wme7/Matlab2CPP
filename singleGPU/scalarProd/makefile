CUDA_INSTALL_PATH := /usr/local/cuda-7.5
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart

all: MAIN

MAIN: CPU_CORE GPU_CORE
	g++ main.o scalarProd.o $(CUDA_LIB) -O3 -o test.run 

CPU_CORE:
	g++ -O3 -c main.cpp 

GPU_CORE:
	nvcc -O3 -arch compute_35 -c scalarProd.cu 

clean:
	rm *.o
	rm *.txt
	rm *.run
