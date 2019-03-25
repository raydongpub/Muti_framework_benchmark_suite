#!/usr/bin/bash
#!/bin/sh
#SBATCH -J oshm-matmul_2_48
#SBATCH -o oshm-matmul_2_48.o%j
#SBATCH -e oshm-matmul_2_48.e%j
#SBATCH -p gpu3 
#SBATCH -N 2 
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431"
#SBATCH --ntasks-per-node=20 --ntasks-per-core=20 --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time mpirun -np 48 --oversubscribe --hostfile host2_48 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset9984.nn -b 64 -t 128
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset4800.nn -b 64 -t 128
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 36 --hostfile host36 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 49 --hostfile host49 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 64 --hostfile host64 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_oshmcuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
