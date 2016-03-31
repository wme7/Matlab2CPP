#!/bin/sh

METHOD=3
EPSILON=3.0
MACHFILE=${1}

if [ -z "${MACHFILE}" ]; then
  echo machinefile not specified 
  exit
fi
if [! -e "${MACHFILE}" ]; then
  echo machinefile not found
  exit
fi

PROCS="1 2 4 8 16 28"
HEIGHT="64 128 256 512" # 1024 2048" # 4096 8192" # min has to be at least max num procs!! (need to fix this)
WIDTH="64 128 256 512" # 1024 2048 4096 8192"
REPEAT=`seq 1 2`                     #"1 2 3"

echo "Method $METHOD Epsilon $EPSILON"
echo "Widths $WIDTH"
echo "Heights $HEIGHT"
echo "Procs $PROCS"
echo
for w in ${WIDTH}; do
  for h in ${HEIGHT}; do
    echo "... W     H   P     H/P          W/H       Time(s)         Spd Up          %Eff"
    serial=0
    for p in ${PROCS}; do
      for interation in ${REPEAT}; do
        out=`mpirun -np ${p} -machinefile ${MACHFILE} ../bin/2dheat.x -t -w ${w} -h ${h} -m ${METHOD} -e ${EPSILON}`
        # capture serial time
        if [ 1 -eq ${p} ]; then
          serial=$out
        fi
        # calculate speed up
        s=`perl -e "print ($serial / $out)"`
        # calculate efficiency
        e=`perl -e "print ($s / $p)"`
        # rows per cpu
        r=`perl -e "print ($h / $p)"`
        # w / h
        c=`perl -e "print ($w / $p)"`
        printf "%5d %5d %3d %10.5f %10.5f %15.9f %15.9f %15.9f\n" $w $h $p $r $c $out $s $e
      done
    done
  done
done
