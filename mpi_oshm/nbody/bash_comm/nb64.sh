#!/usr/bin/bash
#SBATCH -J NB_64
#SBATCH -o NB_64_out
#SBATCH -e NB_64_err
#SBATCH -p GPU -N 10 --mem=100g --ntasks-per-node=2
#sSBATCH -p GPU -w "c11u11 c11u13 c11u15 c11u17" --mem=100g --ntasks-per-node=2
time mpirun -n 64 --hostfile ompihosts -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/ksajjapongse/data/benchmark/nbody/nb_mpicuda -c /home/ksajjapongse/data/benchmark/nbody/mpi.conf
