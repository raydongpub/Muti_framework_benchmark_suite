#!/usr/bin/bash
#!/bin/sh
#SBATCH -J mpi-nw
#SBATCH -o mpi-nw.o%j
#SBATCH -e mpi-nw.e%j
#SBATCH -p gpu3 
#SBATCH -N 2
#SBATCH -w "lewis4-r730-gpu3-node428, lewis4-r730-gpu3-node429"
#SBATCH --ntasks-per-node=8 --mem=100g
#time mpirun -np 4 --hostfile host main.x
time mpirun -np 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2000 -b 16 -t 128
time mpirun -np 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2000 -b 16 -t 256
time mpirun -np 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2000 -b 16 -t 512
time mpirun -np 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2000 -b 32 -t 128
time mpirun -np 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2000 -b 32 -t 256
time mpirun -np 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2000 -b 32 -t 512
time mpirun -np 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2000 -b 64 -t 128
time mpirun -np 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2000 -b 64 -t 256
time mpirun -np 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2000 -b 64 -t 512
