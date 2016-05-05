all: MAIN

MAIN: CORE FUNCS
	g++ main.o heat1d.o -O3 -o heat1d.run

CORE: main.cpp
	g++ -O3 -c main.cpp

FUNCS: heat1d.cpp
	g++ -O3 -c heat1d.cpp

clean:
	rm *.o
	rm *.run
	rm *.txt
