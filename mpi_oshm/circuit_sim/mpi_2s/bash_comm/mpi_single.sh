#!/usr/bin/bash
#!/bin/sh
#SBATCH -J mpi-cct
#SBATCH -o mpi-cct.o%j
#SBATCH -e mpi-cct.e%j
#SBATCH -p gpu3 
#SBATCH -N 1
#SBATCH -w "lewis4-r730-gpu3-node429"
#SBATCH --mem=100g --ntasks-per-node=20
#SBATCH --distribution=cyclic:cyclic
#time mpirun -np 4 main.x
time mpirun -np 7 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 600 -n 100 -w 100 -b 16 -t 256 
time mpirun -np 7 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 1200 -n 100 -w 100 -b 16 -t 256 
time mpirun -np 7 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 2400 -n 100 -w 100 -b 16 -t 256 
time mpirun -np 11 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 600 -n 100 -w 100 -b 16 -t 256 
time mpirun -np 11 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 1200 -n 100 -w 100 -b 16 -t 256 
time mpirun -np 11 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 2400 -n 100 -w 100 -b 16 -t 256 
time mpirun -np 16 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 600 -n 100 -w 100 -b 16 -t 256 
time mpirun -np 16 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 1200 -n 100 -w 100 -b 16 -t 256 
time mpirun -np 16 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 1 -p 2400 -n 100 -w 100 -b 16 -t 256 
