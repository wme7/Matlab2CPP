MPI_LIB := -L /home/manuel/openMPI/include/

all: MAIN

MAIN: check.cpp
	mpic++ $(MPI_LIB) check.cpp -o checkCUDAaware.run

clean:
	rm *.run
