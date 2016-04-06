CUDA_INSTALL_PATH := /usr/local/cuda
CUDA_INCLUDES := -I $(CUDA_INSTALL_PATH)/include
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart

all: GPU_CORE CPU_CORE
	mpiCC main.o gpu_main.o -o test.run $(CUDA_INCLUDES) $(CUDA_LIB)	
CPU_CORE:
	mpiCC main.c -c

GPU_CORE:
	nvcc gpu_main.cu -c
clean:
	rm *.run
