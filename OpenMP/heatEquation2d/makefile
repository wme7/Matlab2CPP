all: MAIN

MAIN: CORE FUNCS
	g++ -Wall -g -fopenmp main.o heat2d.o -o heat2d.run

CORE: main.cpp
	g++ -Wall -g -fopenmp -c main.cpp

FUNCS: heat2d.cpp
	g++ -Wall -g -fopenmp -c heat2d.cpp

clean:
	rm *.txt
	rm *.run
	rm *.o
