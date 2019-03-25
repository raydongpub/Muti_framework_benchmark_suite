#!/usr/bin/bash
#!/bin/sh
#SBATCH -J oshm-himeno
#SBATCH -o oshm-himeno.o%j
#SBATCH -e oshm-himeno.e%j
#SBATCH -p gpu3 
#SBATCH -N 2
#SBATCH -w "lewis4-r730-gpu3-node428, lewis4-r730-gpu3-node429"
#SBATCH --ntasks-per-node=8 --mem=100g
#time mpirun -np 4 --hostfile host main.x
#time mpirun -np 4 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset16.nn
time mpirun -n 8 -x CUDA_VISIBLE_DEVICES=0,1,2,3 bmt_oshmcuda -pe 2 2 2 -ds s
