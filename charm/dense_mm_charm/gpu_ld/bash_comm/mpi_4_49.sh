#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-matmul_4_49
#SBATCH -o charm-matmul_4_49.o%j
#SBATCH -e charm-matmul_4_49.e%j
#SBATCH -p gpu3 
#SBATCH -N 4 --gres=gpu:1
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431, lewis4-r730-gpu3-node432, lewis4-r730-gpu3-node433"
#SBATCH -n 49 -m plane=12 --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time charmrun matmul 10045 10045 5 5 49 /home/rgcxc/data/workspace/benchmark/data_set/dset10045.nn ++nodegroup charm-matmul-ffot +p49 ++mpiexec
#time mpirun -np 49 --hostfile host4_49 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10045.nn -b 64 -t 128
# --ntasks-per-core=49
#--ntasks-per-node=13 --mem=100g
