# Manuel's simplified Makefile for NCCL+MPI *.cu files
# NHRI 2016.04.20

# Compiler and libraries paths
MPI_HOME ?= /home/manuel/openMPI
CUDA_HOME ?= /usr/local/cuda-7.5
NCCL_HOME ?= /home/manuel/nccl

# Cuda code target machine arquitecture
CUDACODE := -gencode=arch=compute_35,code=sm_35 

# Basic Flags
CPPFLAGS   := -I$(CUDA_HOME)/include
CXXFLAGS   := -O3 -fPIC -fvisibility=hidden
NVCUFLAGS  := $(CUDACODE) -O3 -lineinfo -std=c++11 -maxrregcount 96

# ------------------
VERBOSE ?= 0
ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xptxas -v -Xcompiler -Wall,-Wextra
CXXFLAGS  += -Wall -Wextra
endif
# ------------------

# Path to includes and libs
LDFLAGS    := -L$(CUDA_HOME)/lib64 -lcudart
MPIFLAGS   := -I$(MPI_HOME)/include -L$(MPI_HOME)/lib -lmpi
NCCLFLAGS  := -I$(NCCL_HOME)/include -L$(NCCL_HOME)/lib 

# linker for nccl.so 
LIBNAME    := libnccl.so
LIBLINK    := $(patsubst lib%.so, -l%, $(LIBNAME))

all: test 

test : mpi_test.cu 
	@printf "Building  %-25s > %-24s\n" $< $@.run
	nvcc $(NCCLFLAGS) $(MPIFLAGS) $(CPPFLAGS) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" -o $@.run $<  $(LIBLINK) $(LDFLAGS) -lcuda -lcurand -lnvToolsExt

clean:
	rm *.run
