#!/bin/bash
#SBATCH -N 1
#SBATCH -B 2:10
#SBATCH -t 00:10:00
#SBATCH -o result.txt

export I_MPI_PIN_DOMAIN=omp
export OMP_NUM_THREADS=10
export plm_ple_memory_allocation_policy=localalloc


mpirun -np 1 -bind-to socket -npersocket 1 ./a.out
