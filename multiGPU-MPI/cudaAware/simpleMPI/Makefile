CUDA_INSTALL_PATH := /usr/local/cuda-7.5
CUDA_INCLUDES := -I $(CUDA_INSTALL_PATH)/include
CUDA_LIB := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart

all: GPU_CORE CPU_CORE
	mpic++ $(CUDA_INCLUDES) $(CUDA_LIB) cpu_core.o gpu_core.o -o simpleMPI.run 

CPU_CORE: simpleMPI.cpp
	mpic++ -c simpleMPI.cpp -o cpu_core.o

GPU_CORE: simpleMPI.cu
	nvcc -c simpleMPI.cu -o gpu_core.o

clean:
	rm *.run
	rm *.o

