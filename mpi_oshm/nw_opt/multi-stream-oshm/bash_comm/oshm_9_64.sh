#!/usr/bin/bash
#!/bin/sh
#SBATCH -J oshm-nw_9_64
#SBATCH -o oshm-nw_9_64.o%j
#SBATCH -e oshm-nw_9_64.e%j
#SBATCH -p gpu3 
#SBATCH -N 9
#SBATCH -w "lewis4-r730-gpu3-node428, lewis4-r730-gpu3-node429, lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431, lewis4-r730-gpu3-node432, lewis4-r730-gpu3-node433, lewis4-r730-gpu3-node434, lewis4-r730-gpu3-node435, lewis4-r730-gpu3-node476" 
#SBATCH --ntasks-per-node=20 --ntasks-per-core=20 --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time mpirun -np 64 --oversubscribe --hostfile host9_64 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -ds 2 -l 1000 -n 4992 -b 64 -t 512 
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -m /home/rgcxc/data/workspace/benchmark/data_set/dset4800.nn -b 64 -t 128
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 36 --hostfile host36 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 49 --hostfile host109 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 64 --hostfile host64 -x CUDA_VISIBLE_DEVICES=0,1,2,3 nw-single-stream -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
