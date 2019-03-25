#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-himeno_4_64
#SBATCH -o charm-himeno_4_64.o%j
#SBATCH -e charm-himeno_4_64.e%j
#SBATCH -p gpu3 
#SBATCH -N 4 --gres=gpu:1
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431, lewis4-r730-gpu3-node432, lewis4-r730-gpu3-node433"
#SBATCH --ntasks-per-node=16 --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time charmrun himeno -pe 4 4 4 -ds xl -c 64 +p64 ++mpiexec
#time charmrun himeno 10080 10080 5 5 64 /home/rgcxc/data/workspace/benchmark/data_set/dset10080.nn ++nodegroup charm-himeno-ftht +p64 ++mpiexec
#time mpirun -np 64 --hostfile host4_64 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10080.nn -b 64 -t 128
# --ntasks-per-core=64
#--ntasks-per-node=5 --mem=100g
#-n 64 -m plane=12 --mem=100g
