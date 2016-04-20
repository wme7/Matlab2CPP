# Manuel's simplified Makefile for NCCL+MPI *.cu files
# NHRI 2016.04.20

# Compiler and libraries paths
CUDA_HOME ?= /usr/local/cuda-7.5
NCCL_HOME ?= /home/manuel/nccl
MPI_HOME ?= /home/manuel/openMPI

# Basic Flags
CXXFLAGS   := -O3 -fPIC -fvisibility=hidden
NVCUFLAGS  := -gencode=arch=compute_35,code=sm_35 -O3 -lineinfo -std=c++11 -maxrregcount 96

# ------------------
VERBOSE ?= 0
ifneq ($(VERBOSE), 0)
CXXFLAGS  += -Wall -Wextra
NVCUFLAGS += -Xptxas -v -Xcompiler -Wall,-Wextra
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
	nvcc $(NCCLFLAGS) $(MPIFLAGS) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" -o $@.run $<  $(LIBLINK) $(LDFLAGS) -lcuda -lcurand -lnvToolsExt

clean:
	rm *.run
