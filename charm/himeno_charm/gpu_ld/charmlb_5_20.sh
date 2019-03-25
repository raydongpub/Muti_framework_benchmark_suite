time mpirun -np 20 --hostfile host5_20 --mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 himeno -pe 5 2 2 -ds xl -c 20 +balancer RefineLB >nps_result/charm_ld_5_20_xl.txt
#matmul 9984 9984 4 4 20 /home/rgu/workspace/ben_suit/data_set/dset9984.nn
