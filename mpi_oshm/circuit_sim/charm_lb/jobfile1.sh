#!/bin/sh
#SBATCH -J charm-test
#SBATCH -o charm-test.o%j
#SBATCH -e charm-test.e%j
#SBATCH -p GPU
# Set the number of Node and tasks/cpus
#SBATCH -w "r730-node75, r730-node76"
#SBATCH -N 2
#SBATCH --mem=100g --ntasks-per-node=10
#./experiment -f experiment_1.txt
#mpirun -np 3 -x CUDA_VISIBLE_DEVICES=0,1,2,3 --hostfile hosts /home/rgu/data/workspace/charm/dense_mm_charm/gpu/matmul 16 16 2 2 4 /home/rgu/data/workspace/benchmark/data_set/dset16.nn
charmrun ++nodegroup gpu-run /home/rgu/data/workspace/charm/dense_mm_charm/gpu_ld/matmul 16 16 1 4 4 /home/rgu/data/workspace/benchmark/data_set/dset16.nn +p4 # +balancer RefineLB +LBDebug +LBPeriod 0.5 NeighborLB +LBDebug  
#charmrun  /home/rgu/data/workspace/charm/dense_mm_charm/gpu_ld/matmul 10000 10000 5 1 16 /home/rgu/data/workspace/dense_mm/dset10000.nn ++nodegroup gpu-run +balancer RefineLB +LBDebug +LBPeriod 0.5 #NeighborLB +LBDebug  #
#charmrun ++mpiexec --hostfile ompihosts -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/dense_mm_charm/gpu_ld/matmul 10000 10000 5 1 16 /home/rgu/data/workspace/dense_mm/dset10000.nn +p14 +balancer RefineLB +LBDebug +LBPeriod 0.5 #NeighborLB +LBDebug  #

#time mpirun -n 20 --hostfile ompihosts -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/dense_mm_charm/gpu_ld/matmul 10000 10000 1 1 1000 /home/rgu/data/workspace/charm/dense_mm_charm/gpu_ld/dset10000.nn +balancer NeighborLB +LBDebug #+isomalloc_syn
#mpirun -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/dense_mm_charm/gpu/matmul 6000 6000 10 1 40 /home/rgu/data/workspace/charm/dense_mm_charm/gpu/dset6000.nn +balancer GreedyLB +LBDebug #+isomalloc_sync
