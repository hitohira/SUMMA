#!/bin/bash

#PBS -q u-medium
#PBS -l select=32:ncpus=36:mpiprocs=32:ompthreads=1
#PBS -W group_list=gi16
#PBS -l walltime=01:00:00
#PBS -o result.txt
#PBS -e err.txt

cd $PBS_O_WORKDIR
mpiexec -np 1024 ./main

