#!/bin/bash

#PBS -l select=2:ncpus=2:mpiprocs=2:mem=100m,place=scatter
#PBS -l walltime=00:00:10
#PBS -m n
#PBS -o out-clu-demo.txt
#PBS -e err-clu-demo.txt

MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')

cd $PBS_O_WORKDIR

echo "Node file path: $PBS_NODEFILE"
echo "Node file contents:"
cat $PBS_NODEFILE

echo "Using mpirun at which mpirun"
echo "Running $MPI_NP MPI processes"

mpirun -machinefile $PBS_NODEFILE -np $MPI_NP ./main