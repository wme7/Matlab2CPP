make clean
make
mpirun -np 2 heat3d_async.run 256 256 512 100 64 4 1
