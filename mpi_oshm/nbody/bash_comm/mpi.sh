#!/usr/bin/bash
#!/bin/sh
#SBATCH -J mpi-nb
#SBATCH -o mpi-nb.o%j
#SBATCH -e mpi-nb.e%j
#SBATCH -p gpu3 
#SBATCH -N 3
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431"
#SBATCH --ntasks-per-node=20 --mem=100g
#time mpirun -np 4 --hostfile host main.x
time mpirun -np 10 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_mpicuda -c mpi.conf -b 16 -t 128
time mpirun -np 10 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_mpicuda -c mpi.conf -b 16 -t 256
time mpirun -np 10 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_mpicuda -c mpi.conf -b 16 -t 512
time mpirun -np 10 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_mpicuda -c mpi.conf -b 32 -t 128
time mpirun -np 10 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_mpicuda -c mpi.conf -b 32 -t 256
time mpirun -np 10 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_mpicuda -c mpi.conf -b 32 -t 512
time mpirun -np 10 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_mpicuda -c mpi.conf -b 64 -t 128
time mpirun -np 10 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_mpicuda -c mpi.conf -b 64 -t 256
time mpirun -np 10 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_mpicuda -c mpi.conf -b 64 -t 512
