#!/usr/bin/bash
#!/bin/sh
#SBATCH -J legion-matmul
#SBATCH -o legion-matmul.o%j
#SBATCH -e legion-matmul.e%j
#SBATCH -p gpu3 
#SBATCH -N 2 -n 4 
#SBATCH -w "lewis4-r730-gpu3-node432, lewis4-r730-gpu3-node431"
#SBATCH --ntasks-per-node=20 --ntasks-per-core=20 --mem=100g
mpirun -np 2 --oversubscribe --hostfile host -x CUDA_VISIBLE_DEVICES=0,1,2,3 circuit 2 -ll:gpu 1 -ll:cpu 4 
#export GASNET_SSH_SERVERS="lewis4-r730-gpu3-node430 lewis4-r730-gpu3-node431"
#amudprun -np 2 -spawn S circuit
