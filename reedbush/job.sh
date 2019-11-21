#!/bin/bash
#PBS -q u-debug
#PBS -l select=4:mpiprocs=16;ompthreads=2
#PBS -l walltime=00:20:00
#PBS -W group_list=gi16


cd $PBS_O_WORKDIR
mpiexec -np 64 ./a.out
