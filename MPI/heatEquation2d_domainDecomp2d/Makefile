# Paths
MPI_INSTALL_PATH := /home/manuel/openMPI

# Libraries
MPI_INCLUDES := -I $(MPI_INSTALL_PATH)/include

# flags
CFLAGS= -O3 -march=native #-Wall

# Commands
all: MAIN CPU_CORE
	mpic++ $(MPI_INCLUDES) $(CFLAGS) main.o tools2d.o -o heat2d.run

MAIN: main.cpp
	mpic++ $(MPI_INCLUDES) $(CFLAGS) -c main.cpp

CPU_CORE: heat2d.cpp
	mpic++ $(MPI_INCLUDES) $(CFLAGS) -c heat2d.cpp -o tools2d.o

clean:
	rm *.o
	rm *.run
	rm *.txt
