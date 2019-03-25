#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-matmul_64
#SBATCH -o charm-matmul_64.o%j
#SBATCH -e charm-matmul_64.e%j
#SBATCH -p gpu3 
#SBATCH -N 2 
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431"
#SBATCH --ntasks-per-node=32 --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time charmrun matmul 9984 9984 4 4 64 /home/rgcxc/data/workspace/benchmark/data_set/dset9984.nn ++nodegroup charm-matmul-six +p64 ++mpiexec
#time mpirun -np 48 --oversubscribe --hostfile host2_48 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset9984.nn -b 64 -t 128
#--ntasks-per-core=20 
