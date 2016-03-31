#!/bin/sh

for m in 1 2 3; do
  for i in 1 2 3 4 5 6 7 8 9; do 
    for s in static dynamic guided; do
      OUT=`OMP_NUM_THREADS=$i OMP_SCHEDULE=$s ../src/openmp-2dheat.x -w 100 -h 100 -m $m -t` 
      printf "${OUT}\tm=$m\t$s\n"
    done
  done
done
