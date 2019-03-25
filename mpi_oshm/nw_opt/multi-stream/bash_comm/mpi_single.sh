#!/usr/bin/bash
#!/bin/sh
#SBATCH -J mpi-nw
#SBATCH -o mpi-nw.o%j
#SBATCH -e mpi-nw.e%j
#SBATCH -p gpu3 
#SBATCH -N 1
#SBATCH -w "lewis4-r730-gpu3-node429"
#SBATCH --ntasks-per-node=20 --mem=100g
#time mpirun -np 4 main.x
time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 600 -b 64 -t 512
time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 1200 -b 64 -t 512
time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2400 -b 64 -t 512
time mpirun -np 10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 600 -b 64 -t 512
time mpirun -np 10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 1200 -b 64 -t 512
time mpirun -np 10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2400 -b 64 -t 512
time mpirun -np 20 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 600 -b 64 -t 512
time mpirun -np 20 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 1200 -b 64 -t 512
time mpirun -np 20 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 2400 -b 64 -t 512
