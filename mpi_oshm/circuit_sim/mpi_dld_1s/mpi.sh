#!/usr/bin/bash
#!/bin/sh
#SBATCH -J mpi-cct
#SBATCH -o mpi-cct.o%j
#SBATCH -e mpi-cct.e%j
#SBATCH -p gpu3 
#SBATCH -N 2
#SBATCH -w "lewis4-r730-gpu3-node428, lewis4-r730-gpu3-node429"
#SBATCH --ntasks-per-node=8 --mem=100g
#time mpirun -np 4 --hostfile host main.x
time mpirun -np 3 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 4 -n 20 -w 40 -pp 2 -m 4 
