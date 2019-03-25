#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-matmul
#SBATCH -o charm-matmul.o%j
#SBATCH -e charm-matmul.e%j
#SBATCH -p gpu3 
#SBATCH -N 2
#SBATCH -w "lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431"
#SBATCH --ntasks-per-node=20 --mem=100g
#time mpirun -np 4 --hostfile host main.x
#time mpirun -np 4 --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset16.nn
mpirun --hostfile ompihosts matmul 16 16 1 4 4 /home/rgcxc/data/workspace/benchmark/data_set/dset16.nn +p4 ++mpiexec # +balancer RefineLB +LBDebug +LBPeriod 0.5 NeighborLB +LBDebug  
#charmrun ++nodegroup gpu-run matmul 16 16 1 4 4 /home/rgcxc/data/workspace/benchmark/data_set/dset16.nn +p4 ++mpiexec # +balancer RefineLB +LBDebug +LBPeriod 0.5 NeighborLB +LBDebug  
#charmrun --hostfile omphost -x CUDA_VISIBLE_DEVICES=0,1,2,3 matmul 16 16 1 4 4 /home/rgcxc/data/workspace/benchmark/data_set/dset16.nn +p4


#./experiment -f experiment_1.txt
#mpirun -np 3 -x CUDA_VISIBLE_DEVICES=0,1,2,3 --hostfile hosts /home/rgu/data/workspace/charm/dense_mm_charm/gpu/matmul 16 16 2 2 4 /home/rgu/data/workspace/benchmark/data_set/dset16.nn
#charmrun  /home/rgu/data/workspace/charm/dense_mm_charm/gpu_ld/matmul 10000 10000 5 1 16 /home/rgu/data/workspace/dense_mm/dset10000.nn ++nodegroup gpu-run +balancer RefineLB +LBDebug +LBPeriod 0.5 #NeighborLB +LBDebug  #
#charmrun ++mpiexec --hostfile ompihosts -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/dense_mm_charm/gpu_ld/matmul 10000 10000 5 1 16 /home/rgu/data/workspace/dense_mm/dset10000.nn +p14 +balancer RefineLB +LBDebug +LBPeriod 0.5 #NeighborLB +LBDebug  #

#time mpirun -n 20 --hostfile ompihosts -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/dense_mm_charm/gpu_ld/matmul 10000 10000 1 1 1000 /home/rgu/data/workspace/charm/dense_mm_charm/gpu_ld/dset10000.nn +balancer NeighborLB +LBDebug #+isomalloc_syn
#mpirun -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/dense_mm_charm/gpu/matmul 6000 6000 10 1 40 /home/rgu/data/workspace/charm/dense_mm_charm/gpu/dset6000.nn +balancer GreedyLB +LBDebug #+isomalloc_sync
