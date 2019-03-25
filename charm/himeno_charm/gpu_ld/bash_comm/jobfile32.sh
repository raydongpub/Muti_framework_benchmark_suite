#!/bin/sh
#SBATCH -J charm-test32
#SBATCH -o charm-test32.o%j
#SBATCH -e charm-test32.e%j
#SBATCH -p GPU
# Set the number of Node and tasks/cpus
#SBATCH -N 10
#SBATCH --mem=100g --ntasks-per-node=2
#mpirun -n 10 -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/himeno_charm/gpu_ld/himeno -pe 2 2 5 -ds xl -c 20 +balancer RefineLB +LBDebug #+isomalloc_sync

charmrun --hostfile ompihosts  -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/rgu/data/workspace/charm/himeno_charm/gpu_ld/himeno -pe 4 4 2 -ds xl -c 32 +p10 +balancer RefineLB +LBDebug
