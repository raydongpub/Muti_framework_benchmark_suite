#!/usr/bin/bash
#!/bin/sh
#SBATCH -J mpi-himeno
#SBATCH -o mpi-himeno.o%j
#SBATCH -e mpi-himeno.e%j
#SBATCH -p gpu3 
#SBATCH -N 2
#SBATCH -w "lewis4-r730-gpu3-node431, lewis4-r730-gpu3-node430"
#SBATCH --ntasks-per-node=20 --mem=100g
#time mpirun -np 4 --hostfile host main.x
time mpirun -n 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 bmt_mpicuda -pe 2 2 2 -ds xl -b 16 -t 64
#time mpirun -n 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 bmt_mpicuda -pe 2 2 2 -ds xl -b 4 -t 256
#time mpirun -n 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 bmt_mpicuda -pe 2 2 2 -ds xl -b 4 -t 512
#time mpirun -n 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 bmt_mpicuda -pe 2 2 2 -ds xl -b 8 -t 128
#time mpirun -n 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 bmt_mpicuda -pe 2 2 2 -ds xl -b 8 -t 256
#time mpirun -n 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 bmt_mpicuda -pe 2 2 2 -ds xl -b 8 -t 512
#time mpirun -n 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 bmt_mpicuda -pe 2 2 2 -ds xl -b 16 -t 128
#time mpirun -n 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 bmt_mpicuda -pe 2 2 2 -ds xl -b 16 -t 256
#time mpirun -n 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 bmt_mpicuda -pe 2 2 2 -ds xl -b 16 -t 512
