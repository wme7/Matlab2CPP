CUDA_INSTALL_PATH := /usr/local/cuda-7.5
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart -fopenmp -lgomp

all: MAIN

MAIN: CPU_CORE GPU_CORE
	g++ $(CUDA_LIB) -O3 main.o cuda_main.o -o SHLL.run 

CPU_CORE:
	g++ -O3 -fopenmp -lgomp -c main.cpp

GPU_CORE:
	nvcc -O3 -arch compute_35 -Xcompiler -fopenmp -lgomp -c cuda_main.cu 

clean:
	rm *.o
	rm *.run
	rm *.txt




	
