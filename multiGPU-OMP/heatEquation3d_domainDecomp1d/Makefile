# Coded by Manuel A. Diaz
# NHRI, 2016.04.29

# Compilers
NVCC = $(shell which nvcc)
CXX = $(shell which g++)

# CUDA path
CUDA_INSTALL_PATH := /usr/local/cuda-7.5

# Compiler flags
CFLAGS=-m64 -O3 -march=native -Wall -fopenmp -funroll-loops
PTXFLAGS=-v
CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include

# Compute flags
GENCODE_SM35 := -gencode arch=compute_35,code=sm_35
GENCODE_SM30 := -gencode arch=compute_37,code=sm_37
GENCODE_SM20 := -gencode arch=compute_50,code=sm_50
GENCODE_FLAGS := $(GENCODE_SM35) $(GENCODE_SM50)

NVCCFLAGS =-O3 -m64 $(GENCODE_FLAGS) -Xcompiler -fopenmp -Xcompiler -fno-strict-aliasing -Xcompiler -funroll-loops #-Xptxas $(PTXFLAGS)
CUDALDFLAGS = -L${CUDA_INSTALL_PATH}/lib64 -lcudart

# Make instructions
all: heat3d_sync.run heat3d_async.run

Util.o: Util.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
Kernels.o: Kernels.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
Sync.o: Sync.c
	$(CXX) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<

Async.o: Async.c
	$(CXX) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<

Host.o: Host.c
	$(CXX) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<

heat3d_sync.run: Sync.o Host.o Util.o Kernels.o
	$(CXX) -o $@ $+ $(CUDALDFLAGS) $(CFLAGS)

heat3d_async.run: Async.o Host.o Util.o Kernels.o
	$(CXX) -o $@ $+ $(CUDALDFLAGS) $(CFLAGS)
			
clean:
	rm -rf *.vtk *.o *.run
