CUDA_INSTALL_PATH := /usr/local/cuda-7.5
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart 

all: MAIN

MAIN: main.cu
	nvcc $(CUDA_LIB) main.cu -o test.run

clean:
	rm *.run
