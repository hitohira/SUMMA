MPI_COMPILE_FLAGS = $(shell mpicc --showme:compile)
MPI_LINK_FLAGS = $(shell mpicc --showme:link)


mpi:
	mpicc -O2 basic.c

job:
	sbatch job.sh

a:
	mpicc -O2 basic.c;sbatch job.sh

test:
	gcc -O2 $(MPI_COMPILE_FLAGS) omptest.c $(MPI_LINK_FLAGS);mpirun -np 1 ./a.out	
