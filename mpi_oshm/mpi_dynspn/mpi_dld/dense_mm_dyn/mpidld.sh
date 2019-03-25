#!/usr/bin/bash
#!/bin/sh
#SBATCH -J mpidld-matmul
#SBATCH -o mpidld-matmul.o%j
#SBATCH -e mpidld-matmul.e%j
#SBATCH -p gpu3 
#SBATCH -N 2
#SBATCH -w "lewis4-r730-gpu3-node428, lewis4-r730-gpu3-node429"
#SBATCH --ntasks-per-node=20 --mem=100g
#time mpirun -np 4 --hostfile host main.x

#time mpiexec -np 3 -x CUDA_VISIBLE_DEVICES=0,1,2,3 -hostfile host --map-by node -mca orte_base_help_aggregate 0 vecmmcuda -pe 4 -pn 2 -m 16 -pp 2
time mpiexec --hostfile hosts -n 3 -map-by node ./vecmmcuda -pe 4 -pn 2 -m 16 -pp 2
#time mpirun -n 10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/mpi_dynspn/dense_mm_dyn/vecmmcuda -pe 9 -pn 10 -m 900 -pp 2
#time mpirun -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/benchmark/mpi_dynspn/mpi_dld/dense_mm_dyn/vecmmcuda -pe 2 -pn 3 -m 2 -pp 1

