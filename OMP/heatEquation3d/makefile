
all: MAIN

MAIN: CPU_CORE GPU_CORE
	g++ main.o heat3d.o -O3 -fopenmp -lgomp -o heat3d.run

CPU_CORE: main.cpp
	g++ -O3 -c main.cpp -fopenmp -lgomp 

GPU_CORE: heat3d.cpp
	g++ -O3 -c heat3d.cpp -fopenmp -lgomp

clean:
	rm *.o *.txt *.run
