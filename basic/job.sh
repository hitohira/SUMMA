#!/bin/bash
#SBATCH -N 8
#SBATCH -B 2:10
#SBATCH -t 00:10:00
#SBATCH -o result.txt

export OMP_NUM_THREADS=10
export plm_ple_memory_allocation_policy=localalloc


mpirun -np 16 -bind-to-socket -npersocket 1 ./a.out
