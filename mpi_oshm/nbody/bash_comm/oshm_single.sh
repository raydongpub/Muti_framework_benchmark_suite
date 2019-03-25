#!/usr/bin/bash
#!/bin/sh
#SBATCH -J oshm-nb
#SBATCH -o oshm-nb.o%j
#SBATCH -e oshm-nb.e%j
#SBATCH -p gpu3 
#SBATCH -N 1
#SBATCH -w "lewis4-r730-gpu3-node430"
#SBATCH --ntasks-per-node=20 --mem=100g
#time mpirun -np 4 main.x
time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -c oshm1200.conf -b 64 -t 256
time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -c oshm12k.conf -b 64 -t 256
time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -c oshm120k.conf -b 64 -t 256
time mpirun -np 10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -c oshm1200.conf -b 64 -t 256
time mpirun -np 10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -c oshm12k.conf -b 64 -t 256
time mpirun -np 10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -c oshm120k.conf -b 64 -t 256
time mpirun -np 20 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -c oshm1200.conf -b 64 -t 256
time mpirun -np 20 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -c oshm12k.conf -b 64 -t 256
time mpirun -np 20 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -c oshm120k.conf -b 64 -t 256
