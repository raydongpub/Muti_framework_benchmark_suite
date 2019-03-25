#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-nbody_2_48
#SBATCH -o charm-nbody_2_48.o%j
#SBATCH -e charm-nbody_2_48.e%j
#SBATCH -p gpu3 --gres=gpu:1 
#SBATCH -N 2 
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431"
#SBATCH --ntasks-per-node=12 --mem=100g 
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time charmrun nbody -c mpi.conf -s 48 +p48 ++mpiexec
#time charmrun -np 21 --hostfile host2_48 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#--ntasks-per-core=1 
