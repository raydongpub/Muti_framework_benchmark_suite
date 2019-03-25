#time charmrun matmul 10000 10000 5 5 20 /home/rgcxc/data/workspace/benchmark/data_set/dset10k.nn ++nodegroup charm-matmul-ntwe +p20 #+balancer RefineLB
#time mpirun -np 20 --hostfile host5_20 --mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 matmul 10000 10000 5 5 20 /home/rgu/workspace/ben_suit/data_set/dset10k.nn
time mpirun -np 20 --hostfile host5_20 --mca btl self,sm,tcp 10000 10000 5 5 20 /home/rgu/workspace/ben_suit/data_set/dset10k.nn >nps_result/charm_nld_5_20_10k.txt

