MPI_HOME ?= /home/manuel/openMPI
CUDA_HOME ?= /usr/local/cuda-7.5
NCCL_HOME ?= /home/manuel/nccl
VERBOSE ?= 0

CUDACODE := -gencode=arch=compute_35,code=sm_35 

CPPFLAGS   := -I$(CUDA_HOME)/include
CXXFLAGS   := -O3 -fPIC -fvisibility=hidden
NVCUFLAGS  := $(CUDACODE) -O3 -lineinfo -std=c++11 -maxrregcount 96

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xptxas -v -Xcompiler -Wall,-Wextra
CXXFLAGS  += -Wall -Wextra
endif

LDFLAGS    := -L$(CUDA_HOME)/lib64 -lcudart 
MPIFLAGS   := -I$(MPI_HOME)/include -L$(MPI_HOME)/lib -lmpi
NCCLFLAGS  := -I$(NCCL_HOME)/include -L$(NCCL_HOME)/lib

all: TEST

TEST: lib $(TESTBINS)
	nvcc $(NCCLFLAGS) $(CPPFLAGS) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" -o main main.cu -Lbuild/lib $(LIBLINK) $(LDFLAGS) -lcuda -lcurand -lnvToolsExt
	#nvcc $(NCCLFLAGS) $(NCCLLDFLAGS) $(CPPFLAGS) $(NVCUFLAGS) main.cu -lcuda -lcurand -lnvToolsExt


clean:
	rm *.txt
	rm *.o
	rm *.run
