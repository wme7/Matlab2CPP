CFLAGS += -lm -pg
MPICC = mpic++
OMP = -omp

all: serial openmp mpi

serial: serial-2dheat.o
	$(CC) $(CFLAGS) serial-2dheat.o -o serial-2dheat.x 

serial-2dheat.o: serial-2dheat.c 
	$(CC) $(CFLAGS) -c serial-2dheat.c

openmp: openmp-2dheat.o
	$(CC) $(OMP) $(CFLAGS) openmp-2dheat.o -o openmp-2dheat.x 

openmp-2dheat.o: openmp-2dheat.c 
	$(CC) $(CFLAGS) -c openmp-2dheat.c

mpi: mpi-2dheat.o
	$(MPICC) $(CFLAGS) mpi-2dheat.o -o mpi-2dheat.x 

mpi-2dheat.o: mpi-2dheat.c 
	$(MPICC) $(CFLAGS) -c mpi-2dheat.c

clean:
	rm -rf *.o *.x 
