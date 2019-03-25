#!/bin/sh
#SBATCH -J charm-test
#SBATCH -o charm-test.o%j
#SBATCH -e charm-test.e%j
#SBATCH -p GPU
# Set the number of Node and tasks/cpus
#SBATCH -N 10
#TMPSBATCH -w "c11u11, c11u13, c11u15, c11u17, c11u19, c11u21, c12u9, c12u11, c12u13"
#SBATCH --mem=100g --ntasks-per-node=2
#mpirun -np 3 -x CUDA_VISIBLE_DEVICES=0,1,2,3 --hostfile hosts /home/rgu/data/workspace/charm/dense_mm_charm/gpu/matmul 8 8 1 1 4 /home/rgu/data/workspace/charm/dense_mm_charm/gpu/dset8.nn

charmrun --hostfile ompihosts -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/nw/gpu_ld/nw -l 1000 -n 4992 -f 4 -s 64 +p10 +balancer RefineLB +LBDebug +LBPeriod 0.5 #NeighborLB +LBDebug  #

#time mpirun -n 20 --hostfile ompihosts -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/dense_mm_charm/gpu_ld/matmul 10000 10000 1 1 1000 /home/rgu/data/workspace/charm/dense_mm_charm/gpu_ld/dset10000.nn +balancer NeighborLB +LBDebug #+isomalloc_syn
#mpirun -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/dense_mm_charm/gpu/matmul 6000 6000 10 1 40 /home/rgu/data/workspace/charm/dense_mm_charm/gpu/dset6000.nn +balancer GreedyLB +LBDebug #+isomalloc_sync
