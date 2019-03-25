#!/bin/bash

#SBATCH -J mpi_dyn_mm
#SBATCH -o mpitest.o%j
#SBATCH -e mpitest.e%j
#SBATCH -p GPU -N 10 --mem-per-cpu=10G --ntasks-per-node=4

time mpirun -n 10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/mpi_dynspn/dense_mm_dyn/vecmmcuda -pe 9 -pn 10 -m 900 -pp 2

