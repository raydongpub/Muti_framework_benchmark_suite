#!/usr/bin/bash
#SBATCH -p GPU -N 9 -n 20 --mem=100g
time mpirun --hostfile ompihosts -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/benchmark/dense_mm/mm_mpicuda -m /home/rgu/data/workspace/benchmark/dense_mm/dset1.nn
#time mpirun -n 25 --hostfile ompihosts -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/benchmark/dense_mm/mm_mpicuda -m /home/rgu/data/workspace/benchmark/dense_mm/dset1.nn
