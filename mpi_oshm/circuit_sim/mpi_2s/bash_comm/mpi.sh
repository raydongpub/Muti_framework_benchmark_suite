#!/usr/bin/bash
#!/bin/sh
#SBATCH -J mpi-cct
#SBATCH -o mpi-cct.o%j
#SBATCH -e mpi-cct.e%j
#SBATCH -p gpu3 
#SBATCH -N 2
#SBATCH -w "lewis4-r730-gpu3-node428, lewis4-r730-gpu3-node429"
#SBATCH --ntasks-per-node=20 --mem=100g
#time mpirun -np 4 --hostfile host main.x
time mpirun -np 9 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 4000 -n 200 -w 400 -b 16 -t 128 
time mpirun -np 9 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 4000 -n 200 -w 400 -b 16 -t 256 
time mpirun -np 9 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 4000 -n 200 -w 400 -b 16 -t 512 
time mpirun -np 9 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 4000 -n 200 -w 400 -b 32 -t 128 
time mpirun -np 9 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 4000 -n 200 -w 400 -b 32 -t 256 
time mpirun -np 9 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 4000 -n 200 -w 400 -b 32 -t 512 
time mpirun -np 9 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 4000 -n 200 -w 400 -b 64 -t 128 
time mpirun -np 9 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 4000 -n 200 -w 400 -b 64 -t 256 
time mpirun -np 9 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 4000 -n 200 -w 400 -b 64 -t 512 
