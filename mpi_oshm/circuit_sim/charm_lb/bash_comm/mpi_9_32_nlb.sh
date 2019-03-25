#!/usr/bin/bash
#!/bin/sh
#SBATCH -J charm-circuit_simnlb_9_32
#SBATCH -o charm-circuit_simnlb_9_32.o%j
#SBATCH -e charm-circuit_simnlb_9_32.e%j
#SBATCH -p gpu3 --gres=gpu:1
#SBATCH -N 9
#SBATCH -w "lewis4-r730-gpu3-node428, lewis4-r730-gpu3-node429, lewis4-r730-gpu3-node430, lewis4-r730-gpu3-node431, lewis4-r730-gpu3-node432, lewis4-r730-gpu3-node433, lewis4-r730-gpu3-node434, lewis4-r730-gpu3-node435, lewis4-r730-gpu3-node476"
#SBATCH -n 32 -m plane=3 --mem=100g
#time mpirun -np 6 -x CUDA_VISIBLE_DEVICES=0,1,2,3 mm_mpicuda -m /home/rgcxc/data/workspace/benchmark/data_set/dset600.nn -b 64 -t 128
time charmrun circuit_sim 2400 100 150 20 32 +p32 ++mpiexec #+balancer RefineLB 
#time charmrun circuit_sim -l 1000 -n 5000 -f 5 -s 32 -b 64 -t 512 +p32 ++mpiexec
#time charmrun circuit_sim 10080 10080 5 5 32 /home/rgcxc/data/workspace/benchmark/data_set/dset10080.nn ++nodegroup charm-circuit_simnlb-ntwe +p32 ++mpiexec
#--ntasks-per-core=32 
#--ntasks-per-node=2 --mem=100g
