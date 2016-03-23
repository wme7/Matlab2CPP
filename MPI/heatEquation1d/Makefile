CUDA_INSTALL_PATH := /usr/local/cuda-7.5
CUDA_INCLUDES := -I $(CUDA_INSTALL_PATH)/include
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart

all: GPU_CORE CPU_CORE
	mpiCC $(CUDA_INCLUDES) $(CUDA_LIB) main.o heat1d.o -o heat1d.run 	
CPU_CORE:
	mpiCC main.cpp -c

GPU_CORE:
	nvcc heat1d.cu -c
clean:
	rm *.run
	rm *.o
	rm *.txt

