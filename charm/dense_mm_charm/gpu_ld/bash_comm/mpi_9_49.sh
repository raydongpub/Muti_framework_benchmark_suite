#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-matmul_9_49
#SBATCH -o charm-matmul_9_49.o%j
#SBATCH -e charm-matmul_9_49.e%j
#SBATCH -p gpu3 --gres=gpu:1
#SBATCH -N 9
#SBATCH -w "lewis4-r730-gpu3-node428, lewis4-r730-gpu3-node429, lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431, lewis4-r730-gpu3-node432, lewis4-r730-gpu3-node433, lewis4-r730-gpu3-node434, lewis4-r730-gpu3-node435, lewis4-r730-gpu3-node476"
#SBATCH -n 49 -m plane=5  --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time charmrun matmul 10045 10045 5 5 49 /home/rgcxc/data/workspace/benchmark/data_set/dset10045.nn ++nodegroup charm-matmul-ntwe +p49 ++mpiexec +balancer DistributedLB #HybridLB #RefineLB 
#time mpirun -np 49 --hostfile host9_49 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset4900.nn -b 64 -t 128
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 36 --hostfile host36 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 49 --hostfile host49 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 64 --hostfile host64 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#--ntasks-per-core=49 
#--ntasks-per-node=6 --distribution=cyclic --mem=100g
