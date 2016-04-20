# Paths
MPI_INSTALL_PATH := /home/manuel/openMPI

# Libraries
MPI_INCLUDES := -I $(MPI_INSTALL_PATH)/include

# flags
CFLAGS= -O3 -march=native #-Wall

# Commands
all: MAIN CPU_CORE
	mpic++ $(MPI_INCLUDES) $(CFLAGS) main.o tools3d.o -o heat3d.run

MAIN: main.cpp
	mpic++ $(MPI_INCLUDES) $(CFLAGS) -c main.cpp -o main.o

CPU_CORE: heat3d.cpp
	mpic++ $(MPI_INCLUDES) $(CFLAGS) -c heat3d.cpp -o tools3d.o

clean:
	rm *.o
	rm *.run
	rm *.txt
