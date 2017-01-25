
# detect if this is my Linux or Mac computer
OS := $(shell uname)
ifeq ($(OS),Darwin)
	CC = gcc-6
else 
	CC = gcc
endif

all: MAIN

MAIN: CPU_CORE GPU_CORE
	$(CC) -O3 -std=c99 main.o advection2d.o -fopenmp -lgomp -lm -o advection2d.run

CPU_CORE: main.c
	$(CC) -O3 -std=c99 -c main.c -fopenmp -lgomp -lm

GPU_CORE: advection2d.c
	$(CC) -O3 -std=c99 -c advection2d.c -fopenmp -lgomp -lm

clean:
	rm -rf *.o *.txt *.run
