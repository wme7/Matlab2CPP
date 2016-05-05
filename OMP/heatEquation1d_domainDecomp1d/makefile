all: MAIN

MAIN: CORE FUNCS
	g++ -Wall -O3 -g -fopenmp main.o heat1d.o -o heat1d.run

CORE: main.cpp
	g++ -Wall -O3 -g -fopenmp -c main.cpp

FUNCS: heat1d.cpp
	g++ -Wall -O3 -g -fopenmp -c heat1d.cpp

clean:
	rm *.txt
	rm *.run
	rm *.o
