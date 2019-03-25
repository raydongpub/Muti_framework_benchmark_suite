#!/usr/bin/bash
#SBATCH -p GPU -N 10 --ntasks-per-node=2 --mem=100g
#sSBATCH -p GPU --gres=gpu:1 -n12 -N9 --mem=100g
time mpirun -n 64 -x CUDA_VISIBLE_DEVICES=0,1,2,3 /home/ksajjapongse/data/benchmark/himenoBMT_dt/bmt_mpicuda -pe 4 4 4 -ds xl
