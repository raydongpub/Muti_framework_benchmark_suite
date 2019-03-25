#time charmrun matmul 10000 10000 5 5 49 /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn ++nodegroup charm-matmul-ntwe +p49 #+balancer RefineLB
time mpirun -np 49 --hostfile host5_49 matmul 10045 10045 5 5 49 /home/rgu/workspace/ben_suit/data_set/dset10045.nn +balancer RefineLB > nps_result/charm_ld_5_49_10k.txt

