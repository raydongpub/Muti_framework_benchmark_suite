#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-matmul_3_7
#SBATCH -o charm-matmul_3_7.o%j
#SBATCH -e charm-matmul_3_7.e%j
#SBATCH -p gpu3 --gres=gpu:1 
#SBATCH -N 3 
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431, lewis4-r730-gpu3-node432"
#SBATCH -n 7 -m plane=2
#SBATCH --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time charmrun matmul 7 7 1 1 7 /home/rgcxc/data/workspace/benchmark/data_set/dset7.nn ++nodegroup charm-matmul-twe +p7 ++mpiexec
#time charmrun -np 21 --hostfile host3_7 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#--ntasks-per-core=1 
#--ntasks-per-node=3 --distribution=cyclic:block --mem=100g 
