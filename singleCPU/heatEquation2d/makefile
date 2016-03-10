all: MAIN

MAIN: CORE FUNCS
	g++ -Wall -g main.o heat2d.o -o heat2d.run

CORE: main.cpp
	g++ -O3 -g -c main.cpp 

FUNCS: heat2d.cpp
	g++ -O3 -g -c heat2d.cpp

clean:
	rm *.o
	rm *.run
	rm *.txt
