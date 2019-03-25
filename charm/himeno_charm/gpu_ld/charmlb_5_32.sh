time mpirun -np 32 --hostfile host5_32 --mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 himeno -pe 4 4 2 -ds xl -c 32 +balancer RefineLB >nps_result/charm_ld_5_32_xl.txt
#matmul 9984 9984 4 4 32 /home/rgu/workspace/ben_suit/data_set/dset9984.nn
