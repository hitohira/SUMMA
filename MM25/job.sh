#!/bin/bash

#PBS -q TODO
#PBS -l select=4:ncpus=36:mpiprocs=16:ompthreads=1
#PBS -W group_list=TODO
#PBS -l walltime=00:30:00
#PBS -o result.txt
#PBS -e err.txt

cd $PBS_O_WORKDIR
mpiexec -np 64 ./main

