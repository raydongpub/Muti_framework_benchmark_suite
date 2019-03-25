#time charmrun matmul 9984 9984 5 5 64 /home/rgcxc/data/workspace/benchmark/data_set/dset9984.nn ++nodegroup charm-matmul-ntwe +p64 #+balancer RefineLB
time mpirun -np 64 --hostfile host5_64 matmul 9984 9984 4 4 64 /home/rgu/workspace/ben_suit/data_set/dset9984.nn +balancer RefineLB >nps_result/charm_ld_5_64_10k.txt

