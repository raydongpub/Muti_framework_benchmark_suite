#!/usr/bin/bash
#SBATCH -p GPU -N 10 --mem=100g --ntasks-per-node=2

time mpirun -n 64 -x CUDA_VISIBLE_DEVICES=0 --hostfile ompihosts ./nw-single-stream -ds 2 -l 1000 -n 4992

