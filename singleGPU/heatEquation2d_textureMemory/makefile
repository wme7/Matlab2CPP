CUDA_INSTALL_PATH := /usr/local/cuda-7.5
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart

all: MAIN

MAIN: main.cu
	nvcc $(CUDA_LIB) -O3 main.cu -o heat2d_Tex.run

clean:
	rm *.txt
	rm *.run
