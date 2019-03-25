time mpirun -np 48 --hostfile host5_48 --mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 himeno -pe 4 4 3 -ds xl -c 48 +balancer RefineLB >nps_result/charm_ld_5_48_xl.txt
#matmul 9984 9984 4 4 48 /home/rgu/workspace/ben_suit/data_set/dset9984.nn
