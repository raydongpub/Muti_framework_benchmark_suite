#!/bin/sh
#SBATCH -J charm-test
#SBATCH -o charm-test.o%j
#SBATCH -e charm-test.e%j
#SBATCH -p GPU
# Set the number of Node and tasks/cpus
#SBATCH -N 10
#SBATCH --mem=100g --ntasks-per-node=2
#mpirun ./nbody -c mpi.conf #+balancer GreedyLB +LBDebug #+isomalloc_sync
charmrun --hostfile ompihosts -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/nbody_charm/gpu_ld/nbody -c /home/rgu/data/workspace/charm/nbody_charm/gpu_ld/mpi.conf -s 128 +p10 +balancer RefineLB +LBDebug +LBPeriod 0.5 #NeighborLB +LBDebug  #
