#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-matmul_2_49
#SBATCH -o charm-matmul_2_49.o%j
#SBATCH -e charm-matmul_2_49.e%j
#SBATCH -p gpu3 --gres=gpu:1 
#SBATCH -N 2 
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431"
#SBATCH -n 49 -m plane=20 --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time charmrun --oversubscribe matmul 10045 10045 5 5 49 /home/rgcxc/data/workspace/benchmark/data_set/dset10045.nn ++nodegroup charm-matmul-fot +p49 ++mpiexec
#time charmrun -np 21 --hostfile host2_49 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#--ntasks-per-core=1 
#--ntasks-per-node=20 --oversubscribe --mem=100g 

