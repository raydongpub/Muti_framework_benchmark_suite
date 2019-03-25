#!/usr/bin/bash
#!/bin/sh
#SBATCH -J mpi-cct_4_32
#SBATCH -o mpi-cct_4_32.o%j
#SBATCH -e mpi-cct_4_32.e%j
#SBATCH -p gpu3 
#SBATCH -N 4 
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431, lewis4-r730-gpu3-node432, lewis4-r730-gpu3-node433"
#SBATCH --ntasks-per-node=20 --ntasks-per-core=20 --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time mpirun -np 32 --oversubscribe --hostfile host4_32 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -l 10 -p 2400 -n 100 -w 150 -b 16 -t 256 
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset4800.nn -b 64 -t 128
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 32 --hostfile host32 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 49 --hostfile host49 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 64 --hostfile host64 -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuitcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
