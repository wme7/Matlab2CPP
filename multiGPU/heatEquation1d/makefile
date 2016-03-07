CUDA_INSTALL_PATH := /usr/local/cuda-7.5
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart -fopenmp -lgomp

all: MAIN

MAIN: CORE FUNCS
	g++ $(CUDA_LIB) main.o heat1d.o -O3 -o heat1d.run

CORE: main.cpp
	g++ -O3 -fopenmp -lgomp -c main.cpp

FUNCS: heat1d.cu
	nvcc -O3 -arch compute_35 -Xcompiler -fopenmp -lgomp -c heat1d.cu

clean:
	rm *.o
	rm *.run
	rm *.txt
