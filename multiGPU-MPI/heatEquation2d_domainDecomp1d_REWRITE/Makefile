# Coded by Manuel A. Diaz
# NHRI, 2016.04.29
 
# Compilers
NVCC = $(shell which nvcc)
MPICXX = $(shell which mpicxx)

# MPI and CUDA paths

CUDA_INSTALL_PATH := /usr/local/cuda-7.5
MPI_INSTALL_PATH := /home/manuel/openMPI

# Compiler flags
CFLAGS=-m64 -O3 -march=native -Wall -fopenmp -funroll-loops
PTXFLAGS=-v
CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include
MPICFLAGS=-I$(MPI_INSTALL_PATH)/include

# Compute flags
GENCODE_SM35 := -gencode arch=compute_35,code=sm_35
GENCODE_SM37 := -gencode arch=compute_37,code=sm_37
GENCODE_SM50 := -gencode arch=compute_50,code=sm_50
GENCODE_FLAGS := $(GENCODE_SM35) $(GENCODE_SM37) $(GENCODE_SM50)

NVCCFLAGS =-O3 -m64 $(GENCODE_FLAGS) -Xcompiler -fopenmp -Xcompiler -fno-strict-aliasing -Xcompiler -funroll-loops #-Xptxas $(PTXFLAGS)
CUDALDFLAGS = -L${CUDA_INSTALL_PATH}/lib64 -lcudart

# Make instructions
all: heat3d_async.run

Util.o: Util.cu
	$(NVCC) $(MPICFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
Kernels.o: Kernels.cu
	$(NVCC) $(MPICFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
Sync.o: Sync.c
	$(MPICXX) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<

Async.o: Async.c
	$(MPICXX) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<
	 
Host_MPI.o: Host_MPI.c
	$(MPICXX) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<

heat3d_sync.run: Sync.o Host_MPI.o Util.o Kernels.o
	$(MPICXX) -o $@ $+ $(CUDALDFLAGS) $(CFLAGS)

heat3d_async.run: Async.o Host_MPI.o Util.o Kernels.o
	$(MPICXX) -o $@ $+ $(CUDALDFLAGS) $(CFLAGS)
			
clean:
	rm -rf *.vtk *.o *.run *.txt 
