#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-nw_2_32
#SBATCH -o charm-nw_2_32.o%j
#SBATCH -e charm-nw_2_32.e%j
#SBATCH -p gpu3 --gres=gpu:1 
#SBATCH -N 2 
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431"
#SBATCH --ntasks-per-node=16 --mem=100g 
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time charmrun nw -l 1000 -n 4992 -f 4 -s 32 -b 64 -t 512 +p32 ++mpiexec
#time charmrun nw 10000 10000 5 5 32 /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn ++nodegroup charm-nw-twe +p32 ++mpiexec
#time charmrun -np 21 --hostfile host2_32 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#--ntasks-per-core=1 
