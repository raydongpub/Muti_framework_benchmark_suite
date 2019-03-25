#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-circuit_sim_4_64
#SBATCH -o charm-circuit_sim_4_64.o%j
#SBATCH -e charm-circuit_sim_4_64.e%j
#SBATCH -p gpu3 
#SBATCH -N 4 --gres=gpu:1
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431, lewis4-r730-gpu3-node432, lewis4-r730-gpu3-node433"
#SBATCH --ntasks-per-node=16 --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time charmrun circuit_sim 2432 100 150 10 64 +p64 ++mpiexec
#time charmrun circuit_sim -l 1000 -n 5000 -f 5 -s 64 -b 64 -t 512 +p64 ++mpiexec
#time charmrun circuit_sim 10000 10000 5 5 64 /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn ++nodegroup charm-circuit_sim-twe +p64 ++mpiexec
#time mpirun -np 64 --hostfile host4_64 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
# --ntasks-per-core=64
