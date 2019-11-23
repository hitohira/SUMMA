#!/bin/bash

#PBS -q TODO
#PBS select=4:ncpus=32:mpiprocs=16:ompthreads=2
#PBS -W group_list=TODO
#PBS -l walltime=00:30:00
#PBS -o result.txt
#PBS -e err.txt

cd $PBS_O_WORKDIR
mpiexec -np 128 ./main

