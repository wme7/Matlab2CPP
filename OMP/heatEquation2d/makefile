
all: MAIN

MAIN: CPU_CORE GPU_CORE
	g++ main.o heat2d.o -O3 -fopenmp -lgomp -o heat2d.run

CPU_CORE: main.cpp
	g++ -O3 -c main.cpp -fopenmp -lgomp 

GPU_CORE: heat2d.cpp
	g++ -O3 -c heat2d.cpp -fopenmp -lgomp

clean:
	rm *.o *.txt *.run
