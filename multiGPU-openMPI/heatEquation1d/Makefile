CUDA_INSTALL_PATH := /usr/local/cuda
CUDA_INCLUDES := -I $(CUDA_INSTALL_PATH)/include
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart

all: GPU_CORE CPU_CORE
	mpiCC main.o heat1d.o -o heat1d.run $(CUDA_INCLUDES) $(CUDA_LIB)	
CPU_CORE:
	mpiCC main.cpp -c

GPU_CORE:
	nvcc heat1d.cu -c
clean:
	rm *.run
