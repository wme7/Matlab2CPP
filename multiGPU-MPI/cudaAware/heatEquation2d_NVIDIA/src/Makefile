
# Compilers
MPICC=$(PREP) mpicc
MPILD=$(PREP) mpic++
NVCC=$(PREP) nvcc

CUDA_INSTALL_PATH := /usr/local/cuda-7.5
MPI_HOME := /home/manuel/openMPI

# Flags
CFLAGS=-std=c99 -O3 -march=native -Wall
MPICFLAGS=-I${MPI_HOME}/include
CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include

GENCODE_SM13    := -gencode arch=compute_13,code=sm_13
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM35    := -gencode arch=compute_37,code=sm_37
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM52    := -gencode arch=compute_52,code=\"sm_52,compute_52\"
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM37) $(GENCODE_SM50) $(GENCODE_SM52)

NVCCFLAGS=-O3 $(GENCODE_FLAGS) -Xcompiler -march=native

CUDALDFLAGS=-L${CUDA_INSTALL_PATH}/lib64 -lcuda -lcudart

# Description of binaries
BINDIR=../bin
JACOBI_CUDA_NORMAL_MPI=$(BINDIR)/jacobi_cuda_normal_mpi
JACOBI_CUDA_AWARE_MPI=$(BINDIR)/jacobi_cuda_aware_mpi
BINARIES=$(JACOBI_CUDA_NORMAL_MPI) $(JACOBI_CUDA_AWARE_MPI)

# Commands
all: $(BINARIES)

device.o: Jacobi.h Device.cu Makefile
	$(NVCC) $(MPICFLAGS) $(NVCCFLAGS) -c Device.cu -o device.o

input.o: Jacobi.h Input.c Makefile
	$(MPICC) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -c Input.c -o input.o

jacobi.o: Jacobi.h Jacobi.c Makefile
	$(MPICC) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -c Jacobi.c -o jacobi.o

host.o: Jacobi.h Host.c Makefile
	$(MPICC) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -c Host.c -o host.o

cuda_normal_mpi.o: Jacobi.h CUDA_Normal_MPI.c Makefile
	$(MPICC) $(MPICFLAGS) $(CFLAGS) $(CUDACFLAGS) -c CUDA_Normal_MPI.c -o cuda_normal_mpi.o

cuda_aware_mpi.o: Jacobi.h CUDA_Aware_MPI.c Makefile
	$(MPICC) $(MPICFLAGS) $(CFLAGS) $(CUDACFLAGS) -c CUDA_Aware_MPI.c -o cuda_aware_mpi.o

$(JACOBI_CUDA_NORMAL_MPI): jacobi.o input.o host.o device.o cuda_normal_mpi.o Makefile
	mkdir -p $(BINDIR)
	$(MPILD) $(CUDALDFLAGS) -o $(JACOBI_CUDA_NORMAL_MPI) jacobi.o input.o host.o device.o  cuda_normal_mpi.o  

$(JACOBI_CUDA_AWARE_MPI): jacobi.o input.o host.o device.o cuda_aware_mpi.o Makefile
	mkdir -p $(BINDIR)
	$(MPILD) $(CUDALDFLAGS) -o $(JACOBI_CUDA_AWARE_MPI) jacobi.o input.o host.o device.o cuda_aware_mpi.o  

doc: CUDA_Normal_MPI.c CUDA_Aware_MPI.c Device.cu Host.c Input.c Makefile Jacobi.c Jacobi.h Jacobi.doxygen
	doxygen Jacobi.doxygen

clean:
	rm -rf *.o *~ $(BINARIES)
