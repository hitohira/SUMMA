#!/bin/bash

#PBS -q u-large
#PBS -l select=36:ncpus=36:mpiprocs=16:ompthreads=0
#PBS -W group_list=TODO
#PBS -l walltime=00:10:00
#PBS -o SUMMA36prof.txt
#PBS -e err.txt

cd $PBS_O_WORKDIR
#module load intel-itac/9.1.2.024
#mpirun -trace -n 256 ./main
mpirun -n 576 ./main

