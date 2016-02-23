CUDA_INSTALL_PATH := /usr/local/cuda-7.5
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/targets/x86_64-linux/lib -lcuda -lcudart -fopenmp -lgomp

all: MAIN

MAIN: CPU_CORE GPU_CORE
	g++ main.o cuda_main.o $(CUDA_LIB) -O3 -o SHLL.run 

CPU_CORE:
	g++ -O3 -c main.cpp

GPU_CORE:
	nvcc -O3 -arch compute_37 -c cuda_main.cu 

clean:
	rm *.o
	rm *.txt
	rm *.run




	
