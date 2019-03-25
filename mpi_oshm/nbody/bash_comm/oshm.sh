#!/usr/bin/bash
#!/bin/sh
#SBATCH -J oshm-nb
#SBATCH -o oshm-nb.o%j
#SBATCH -e oshm-nb.e%j
#SBATCH -p gpu3 
#SBATCH -N 2
#SBATCH -w "lewis4-r730-gpu3-node428, lewis4-r730-gpu3-node430"
#SBATCH --ntasks-per-node=20 --mem=100g
#time mpirun -np 4 --hostfile host main.x
time mpirun -np 8 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -c oshm.conf
