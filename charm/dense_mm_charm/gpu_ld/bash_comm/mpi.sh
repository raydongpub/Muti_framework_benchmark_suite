#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-matmul
#SBATCH -o charm-matmul.o%j
#SBATCH -e charm-matmul.e%j
#SBATCH -p gpu3 --gres=gpu:1
#SBATCH -N 2
#SBATCH -w "lewis4-r730-gpu3-node430 lewis4-r730-gpu3-node431"
#SBATCH --ntasks-per-node=2 --mem=100g
#time charmrun -np 4 --hostfile host main.x
time charmrun matmul 16 16 2 2 4 /home/rgcxc/data/workspace/benchmark/data_set/dset16.nn ++nodegroup charm-matmul +p4 ++mpiexec  
#time mpirun -np 5 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 matmul 16 16 2 2 4 /home/rgcxc/data/workspace/benchmark/data_set/dset16.nn
#--ntasks-per-core=1 
