#!/bin/bash

#SBATCH -J mpi_dyn_ccs
#SBATCH -o mpiwork.o%j
#SBATCH -e mpiwork.e%j
#SBATCH -w "lewis4-r730-gpu3-node426, lewis4-r730-gpu3-node428"
#SBATCH -p gpu3 
#SBATCH -N 2 
#SBATCH --mem-per-cpu=10G 
#SBATCH --ntasks-per-node=10

time mpirun -n 3 --map-by node -x CUDA_VISIBLE_DEVICES=0,1,2,3 --mca orte_base_help_aggregate 0 -hostfile host circuitcuda -l 1 -p 15 -n 10 -w 20 -pp 2 -m 5
#time mpirun -n 3 --map-by node -x CUDA_VISIBLE_DEVICES=0,1,2,3 --mca orte_base_help_aggregate 0 -hostfile host circuitcuda -l 1 -p 16 -n 10 -w 20 -pp 2 -m 4

#time mpirun -n 10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/mpi_dynspn/dense_mm_dyn/vecmmcuda -pe 9 -pn 10 -m 900 -pp 2
#time mpirun -np 3 -x CUDA_VISIBLE_DEVICES=0,1,2,3 -hostfile host --map-by node -mca orte_base_help_aggregate 0 /home/rgu/data/workspace/benchmark/mpi_dynspn/mpi_dld/dense_mm_dyn/vecmmcuda -pe 4 -pn 3 -m 16 -pp 2
#time mpirun -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/benchmark/mpi_dynspn/mpi_dld/dense_mm_dyn/vecmmcuda -pe 2 -pn 3 -m 2 -pp 1

