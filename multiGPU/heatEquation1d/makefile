CUDA_INSTALL_PATH := /usr/local/cuda-7.5
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/targets/x86_64-linux/lib -lcuda -lcudart -fopenmp -lgomp

all: MAIN

MAIN: CORE FUNCS
	g++ main.o heat1d.o $(CUDA_LIB) -O3  -o heat1d.run

CORE: main.cpp
	g++ -O3 -c main.cpp

FUNCS: heat1d.cu
	NVCC -O3 -arch compute_37 -c heat1d.cu

clean:
	rm *.o
	rm *.run
	rm *.txt
