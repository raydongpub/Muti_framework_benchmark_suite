#!/usr/bin/bash
#!/bin/sh
#SBATCH -J oshm-matmul
#SBATCH -o oshm-matmul.o%j
#SBATCH -e oshm-matmul.e%j
#SBATCH -p gpu3 
#SBATCH -N 2
#SBATCH -w "lewis4-r730-gpu3-node428, lewis4-r730-gpu3-node429"
#SBATCH --ntasks-per-node=8 --mem=100g
#time mpirun -np 4 --hostfile host main.x
time mpirun -np 4 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset16.nn -b 32 -t 512
