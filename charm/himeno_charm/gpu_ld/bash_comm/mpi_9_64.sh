#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-himeno_9_64
#SBATCH -o charm-himeno_9_64.o%j
#SBATCH -e charm-himeno_9_64.e%j
#SBATCH -p gpu3 --gres=gpu:1
#SBATCH -N 9
#SBATCH -w "lewis4-r730-gpu3-node428, lewis4-r730-gpu3-node429, lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431, lewis4-r730-gpu3-node432, lewis4-r730-gpu3-node433, lewis4-r730-gpu3-node434, lewis4-r730-gpu3-node435, lewis4-r730-gpu3-node476"
#SBATCH -n 64 -m plane=7 --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time charmrun himeno -pe 4 4 4 -ds xl -c 64 +p64 ++mpiexec +balancer RefineLB #HybridLB 
#time charmrun himeno 10080 10080 5 5 64 /home/rgcxc/data/workspace/benchmark/data_set/dset10080.nn ++nodegroup charm-himeno-ntwe +p64 ++mpiexec
#time mpirun -np 64 --hostfile host9_64 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset4800.nn -b 64 -t 128
#time mpirun -np 10 --hostfile host10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 36 --hostfile host36 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 49 --hostfile host49 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#time mpirun -np 64 --hostfile host64 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn -b 64 -t 128
#--ntasks-per-core=64 
#--ntasks-per-node=2 --mem=100g
