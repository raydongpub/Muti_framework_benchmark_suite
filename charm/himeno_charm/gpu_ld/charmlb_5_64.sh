time mpirun -np 64 --hostfile host5_64 himeno -pe 4 4 4 -ds xl -c 64 +balancer RefineLB >nps_result/charm_ld_5_64_xl.txt
#matmul 9984 9984 4 4 64 /home/rgu/workspace/ben_suit/data_set/dset9984.nn
