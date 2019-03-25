#time charmrun matmul 10080 10080 5 5 36 /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn ++nodegroup charm-matmul-ntwe +p36 #+balancer RefineLB
time mpirun -np 36 --hostfile host5_36 matmul 10080 10080 5 5 36 /home/rgu/workspace/ben_suit/data_set/dset10080.nn >nps_result/charm_nld_5_36_10k.txt

