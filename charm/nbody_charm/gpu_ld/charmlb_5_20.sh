time mpirun -np 20 --hostfile host5_20 --mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 nbody -c mpi.conf -s 20 +balancer RefineLB >nps_result/charm_ld_5_20_1200k.txt

