#!/usr/bin/bash
#!/bin/sh
#SBATCH -J oshm-nb_2_48
#SBATCH -o oshm-nb_2_48.o%j
#SBATCH -e oshm-nb_2_48.e%j
#SBATCH -p gpu3 
#SBATCH -N 2 
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431"
#SBATCH --ntasks-per-node=20 --ntasks-per-core=20 --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time mpirun -np 48 --oversubscribe --hostfile host2_48 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -c oshm1200k.conf -b 64 -t 256 
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset4800.nn -b 64 -t 128
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 36 --hostfile host36 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 49 --hostfile host49 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 64 --hostfile host64 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nb_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
