#!/usr/bin/bash
#SBATCH -p GPU -N 5 --mem=100g --ntasks-per-node=2
#sSBATCH -p GPU -w "c11u11 c11u13 c11u15 c11u17" --mem=100g --ntasks-per-node=2
#time mpirun -n 5 --hostfile ompihosts -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/ksajjapongse/data/benchmark/nbody/nb_mpicuda -c /home/ksajjapongse/data/benchmark/nbody/mpi.conf
time mpirun -n 5 -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/ksajjapongse/data/benchmark/nbody/nb_mpicuda -c /home/ksajjapongse/data/benchmark/nbody/mpi.conf
