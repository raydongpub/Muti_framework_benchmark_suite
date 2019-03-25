#time charmrun matmul 10000 10000 5 5 20 /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn ++nodegroup charm-matmul-ntwe +p20 #+balancer RefineLB
time mpirun -np 20 --hostfile host5_20 matmul 10000 10000 5 5 20 /home/rgu/workspace/ben_suit/data_set/dset10k.nn +balancer RefineLB > nps_result/charm_ld_5_20_10k.txt

